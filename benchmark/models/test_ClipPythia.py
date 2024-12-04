import torch
from transformers import AutoTokenizer, CLIPImageProcessor

from benchmark.models import get_image
from src.model.clip_pythia.model import ClipPythiaForCausalLM
from src.utils.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_IMAGE_TOKEN
from src.utils.conversation import conv_templates, SeparatorStyle


def get_conv():
    template_name = "v1"
    return conv_templates[template_name].copy()


def load_model(model_path, model_name, device='cpu'):
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ClipPythiaForCausalLM.from_pretrained(model_path)

    # get image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    vision_tower = model.get_model().vision_tower

    vision_tower.to(device=device)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.to(device=device)

    return tokenizer, model, image_processor, context_len, num_patches

class TestClipPythia:
    def __init__(self,args,device):
        self.tokenizer, self.model, self.image_processor, self.context_len, self.num_patches = load_model(args.model_path, args.model_name, device)
        self.conv = get_conv()
        self.image_process_mode = "Resize"  # Crop, Resize, Pad
        self.move_to_device()

    def move_to_device(self):
        self.dtype = torch.float32
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        vision_tower = self.model.get_model().vision_tower
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        conv = self.conv.copy()
        text = question + '\n<image>'
        text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        output = \
        self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

        return output

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        images, prompts = [], []
        for image, question in zip(image_list, question_list):
            image = get_image(image)
            conv = self.conv.copy()
            text = question + '\n<image>'
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            images.append(image)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        outputs = self.do_generate(prompts, images, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)

        return outputs

    def get_images_input_ids(self, images, prompts, dtype=torch.float16, keep_aspect_ratio=False):
        if keep_aspect_ratio:
            new_images = []
            for image, prompt in zip(images, prompts):
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False,
                                                        size={"shortest_edge": shortest_edge})['pixel_values'][0]
                new_images.append(image.to(self.model.device, dtype=dtype))

                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.num_patches
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
            images = new_images
        else:
            images = self.image_processor(images, return_tensors='pt')['pixel_values']
            images = images.to(self.model.device, dtype=dtype)
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.num_patches
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompts = [prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token) for prompt in prompts]

        input_ids = self.tokenizer(prompts).input_ids
        batch_size = len(input_ids)
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]

        return images, input_ids, batch_size

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0.2, max_new_tokens=256, stop_str=None,
                    keep_aspect_ratio=False):
        images, input_ids, batch_size = self.get_images_input_ids(images, prompts, dtype, keep_aspect_ratio)

        stop_idx = None
        if stop_str is not None:
            stop_idx = self.tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        output_ids = []
        get_result = [False for _ in range(batch_size)]
        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                    torch.as_tensor(input_ids).to(self.model.device),
                    use_cache=True,
                    images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = self.model(input_ids=token,
                                 use_cache=True,
                                 attention_mask=torch.ones(batch_size, past_key_values[0][0].shape[-2] + 1,
                                                           device=self.model.device),
                                 past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[:, -1]
            if temperature < 1e-4:
                token = torch.argmax(last_token_logits, dim=-1)
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            token = token.long().to(self.model.device)

            output_ids.append(token)
            for idx in range(len(token)):
                if token[idx] == stop_idx or token[idx] == self.tokenizer.eos_token_id:
                    get_result[idx] = True
            if all(get_result):
                break

        output_ids = torch.cat(output_ids, dim=1).long()
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if stop_str is not None:
            for i in range(len(outputs)):
                pos = outputs[i].rfind(stop_str)
                if pos != -1:
                    outputs[i] = outputs[i][:pos]

        return outputs