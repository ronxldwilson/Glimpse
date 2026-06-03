import torch
import numpy as np
import open_clip
from PIL import Image


class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        pil_images = [Image.fromarray(cv2_to_rgb(img)) for img in images]
        tensors = torch.stack([self.preprocess(img) for img in pil_images])
        tensors = tensors.to(self.device)
        features = self.model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def cv2_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1].copy() if img.ndim == 3 else img
