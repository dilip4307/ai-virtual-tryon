
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F
from networks.gmm import GMM

st.set_page_config(page_title="AI Virtual Try-On", layout="centered")
st.title("🛍️ AI-Powered Virtual Try-On System")

uploaded_person = st.file_uploader("Upload person image", type=["jpg", "png"])
uploaded_cloth = st.file_uploader("Upload clothing image", type=["jpg", "png"])

if uploaded_person and uploaded_cloth:
    person_img = Image.open(uploaded_person).convert("RGB")
    cloth_img = Image.open(uploaded_cloth).convert("RGB")

    st.image([person_img, cloth_img], caption=["Person", "Cloth"], width=200)

    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    person = transform(person_img).unsqueeze(0).contiguous()
    cloth = transform(cloth_img).unsqueeze(0).contiguous()

    model = GMM()
    model.eval()

    with torch.no_grad():
        grid, theta = model(person, cloth)
        warped = F.grid_sample(cloth, grid, padding_mode='border')

    output_path = "result/tryon.jpg"
    save_image(warped, output_path)
    st.success("✅ Try-on image generated!")
    st.image(output_path, caption="Try-On Result", use_column_width=True)
else:
    st.info("Please upload both images to proceed.")
