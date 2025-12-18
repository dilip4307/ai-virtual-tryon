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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    person = transform(person_img).unsqueeze(0)
    cloth = transform(cloth_img).unsqueeze(0)

    st.write("Person tensor shape:", person.shape)
    st.write("Cloth tensor shape:", cloth.shape)

    model = GMM(input_nc=3)
    model.eval()

    with torch.no_grad():
        grid, theta = model(person, cloth)
        warped = F.grid_sample(
            cloth,
            grid,
            padding_mode='border',
            align_corners=False
        )

    output_path = "tryon_result.jpg"
    # denormalize for saving
    save_image((warped + 1) / 2, output_path)

    st.success("✅ Try-on image generated")
    st.image(output_path, caption="Try-On Result", use_column_width=True)

    st.info(
        "ℹ️ This GMM variant uses RGB-only inputs and performs global affine warping. "
        "Results are limited compared to modern pose-aware or diffusion-based try-on models."
    )

else:
    st.info("Please upload both images to proceed.")
