import os
from autodistill_paligemma import PaliGemma
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

def test():
  base_model = PaliGemma(
    ontology=CaptionOntology(
        {
            "person": "person",
        }
    ),
    model_id="paligemma-3b-mix-448"
  )
  base_model.model.initialize_pretrained_model()

  # label a single image
  results = base_model.predict(input="image.jpeg")
  plot(
      image=cv2.imread("image.jpeg"),
      classes=base_model.ontology.classes(),
      detections=results
  )

def finetune():
  os.environ["HF_ACCESS_TOKEN"]="hf_xxx"
  from autodistill_paligemma.paligemma_model import PaliGemmaTrainer

  target_model = PaliGemmaTrainer()
  target_model.train("./test.v2i.paligemma/")

def predict_lora():
  device = "cuda:0"
  base_model = PaliGemma(
    ontology=CaptionOntology(
        {
            "person": "person",
        }
    ),
    model_id="paligemma-3b-mix-448"
  )
  from transformers import PaliGemmaForConditionalGeneration
  from peft import LoraConfig
  from peft.peft_model import PeftModel
  lora_config = LoraConfig.from_pretrained("./paligemma-lora", device_map=device  )
  model = (PaliGemmaForConditionalGeneration.from_pretrained( "/tmp/cache/paligemma-pretrains/20/", device_map=device, token="HUGGINGFACETOEKN",).eval().to(device))
  model = PeftModel.from_pretrained(model, "./paligemma-lora").eval().to(device)
  base_model.model.set_model(model)

  import glob
  images = glob.glob("*.jpg")
  for image in images:
    # label a single image
    results = base_model.predict(input=image)
    plot(
        image=cv2.imread(image),
        classes=base_model.ontology.classes(),
        detections=results
    )
