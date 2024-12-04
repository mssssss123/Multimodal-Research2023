import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPImageProcessor, GPTNeoXForCausalLM

from benchmark.models import get_image
from src.utils.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_IMAGE_TOKEN, DIS_IMG_TOKEN
from src.utils.conversation import conv_templates, SeparatorStyle

transform_config = "/data2/meisen/nips2024/src/model/discreate/configs/transform/clip_transform.yaml"
image_tokenizer_config = "/data2/meisen/nips2024/src/model/discreate/configs/tokenizer/seed_llama_tokenizer.yaml"

def get_conv():
    template_name = "v1"
    return conv_templates[template_name].copy()


def load_model(model_path, model_name, device='cpu'):
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPTNeoXForCausalLM.from_pretrained(model_path)

    # get image processor

    tokenizer_cfg = OmegaConf.load(image_tokenizer_config)
    image_tokenizer = hydra.utils.instantiate(tokenizer_cfg, device='cuda', load_diffusion=False)

    transform_cfg = OmegaConf.load(transform_config)
    transform = hydra.utils.instantiate(transform_cfg)


    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.to(device=device)

    return tokenizer, model, transform, image_tokenizer

class TestPythia:
    def __init__(self,args,device):
        self.tokenizer, self.model, self.transform, self.image_tokenizer = load_model(args.model_path, args.model_name, device)
        self.conv = get_conv()
        self.image_process_mode = "Resize"  # Crop, Resize, Pad
        self.move_to_device()

    def move_to_device(self):
        self.dtype = torch.float32

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

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
            image = self.transform(image).to('cuda')
            images.append(image)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        outputs = self.do_generate(prompts, images, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)

        return outputs

    def get_images_input_ids(self, images, prompts, dtype=torch.float16, keep_aspect_ratio=False):
        new_prompts = []
        for image, prompt in zip(images, prompts):
            image_ids = self.image_tokenizer.encode_image(image_torch=image).tolist()[0]
            replace_token = ''.join([DIS_IMG_TOKEN.format(int(item)) for item in image_ids])
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            new_prompts.append(prompt)

        input_ids = self.tokenizer(new_prompts).input_ids
        batch_size = len(input_ids)
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]

        return input_ids, batch_size

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0.2, max_new_tokens=256, stop_str=None,
                    keep_aspect_ratio=False):
        input_ids, batch_size = self.get_images_input_ids(images, prompts, dtype, keep_aspect_ratio)

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
                    use_cache=True,)
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