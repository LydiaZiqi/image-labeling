import streamlit as st
import os
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.client.model import Model 
import cv2
import numpy as np
from urllib.request import urlopen

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

st.title("Data Labeling using Object Detection and GPT4 Vision")


def main():

  IMAGE_URL = st.text_input("Enter the URL of the image to be labeled:", value="https://s3.amazonaws.com/samples.clarifai.com/car.png")

  with st.sidebar:
    st.subheader('Add your Clarifai PAT.')
    clarifai_pat = st.text_input("Clarifai PAT:", type="password")
  if not clarifai_pat:
    st.warning("Please provide your Clarifai PAT to proceed.")
  else:
    os.environ['CLARIFAI_PAT'] = clarifai_pat

    detector_model = Model("https://clarifai.com/clarifai/main/models/objectness-detector")

    prediction_response = detector_model.predict_by_url(IMAGE_URL, input_type='image')

    regions = prediction_response.outputs[0].data.regions

    model_url = 'https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision'
    classes = ['Ferrari 812', 'Volkswagen Beetle', 'BMW M5', 'Honda Civic', 'Alfa Romeo 8C']
    threshold = 0.99 

    response = requests.get(IMAGE_URL)
    img = Image.open(BytesIO(response.content))

    draw = ImageDraw.Draw(img)

    for region in regions:
      top_row = round(region.region_info.bounding_box.top_row, 3)
      left_col = round(region.region_info.bounding_box.left_col, 3)
      bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
      right_col = round(region.region_info.bounding_box.right_col, 3)


      for concept in region.data.concepts:
        prompt = f"Label the Car in the Bounding Box region :({top_row},{left_col},{bottom_row},{right_col} with one word {classes}."
    
        inference_params = dict(temperature=0.2, max_tokens=100, image_url=IMAGE_URL)

        model_prediction = Model(model_url).predict_by_bytes(prompt.encode(), input_type='text', inference_params = inference_params)

        concept_name = model_prediction.outputs[0].data.text.raw
        value = round(concept.value, 4)

        if value > threshold:
          top_row = top_row * img.height
          left_col = left_col * img.width
          bottom_row = bottom_row * img.height
          right_col = right_col * img.width

          draw.rectangle([(int(left_col), int(top_row)), (int(right_col), int(bottom_row))], outline=(36, 255, 12), width=2)

          #display text
          font = ImageFont.load_default()
          draw.text((int(left_col), int(top_row - 10)), concept_name, font=font, fill=(36, 255, 12))

    st.image(img, caption="Labeled Image", channels="BGR", use_column_width=True)


if __name__ == '__main__':
  main()