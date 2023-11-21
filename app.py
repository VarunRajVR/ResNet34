import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ResNet34
import numpy as np

model_path = "weights.pth"
checkpoint = torch.load(model_path, map_location='cpu') 
model = ResNet34()
model.load_state_dict(checkpoint)
model.eval()


def predict(image):
    image = transform(image)
    image = image.unsqueeze(0)  
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.argmax(output).item()
    return predicted_class


# preprocessing
mean = [0.0220, 0.0220, 0.0220]
std = [0.0396, 0.0396, 0.0396]
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0) if x.shape[0] == 1 else x), 
    transforms.Normalize(mean=mean, std=std),
])

class_to_char = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
}


def main():
    st.title("ResNet34 Demo")

    uploaded_image = st.file_uploader("Upload pic", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Given Image", use_column_width=True)
        image = Image.open(uploaded_image)
        preprocessed_image = transform(image)
        preprocessed_image_np = preprocessed_image.permute(1, 2, 0).numpy()  
        preprocessed_image_np = np.clip(preprocessed_image_np, 0.0, 1.0)  

        st.subheader("Preprocessed Image:")
        st.image(preprocessed_image_np, caption="Resized and Normalized Image", use_column_width=True, channels="RGB")

        if st.button("Predict"):
            try:
                result = predict(image)
                predicted_char = class_to_char.get(result, 'Unknown')
                st.success(f"Prediction: Class {result}, Character: {predicted_char}")
            except Exception as e:
                st.error(f" {str(e)}")

if __name__ == "__main__":
    main()