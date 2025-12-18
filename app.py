import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F
from networks.gmm import GMM

# -----------------------------
# NOTE:
# GMM REQUIRES person input with pose + parsing channels.
# This app adds DUMMY pose/parsing maps so the model can run.
# The visual quality will be poor until real pose & parsing
# models are integrated.
# -----------------------------

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

    # RGB tensors
    person_rgb = transform(person_img).unsqueeze(0)
    cloth = transform(cloth_img).unsqueeze(0)

    # -------------------------------------------------
    # DUMMY pose + parsing (PLACEHOLDER)
    # -------------------------------------------------
    # Typical lightweight GMM expects 7 channels total
    # RGB (3) + pose (2) + parsing (2)

    B, _, H, W = person_rgb.shape

    dummy_pose = torch.zeros(B, 2, H, W)
    dummy_parsing = torch.zeros(B, 2, H, W)

    # Final person tensor: [1, 7, 256, 192]
    person = torch.cat([person_rgb, dummy_pose, dummy_parsing], dim=1)

    st.write("Person tensor shape:", person.shape)
    st.write("Cloth tensor shape:", cloth.shape)

    # Load model
    model = GMM()
    model.eval()

    with torch.no_grad():
        grid, theta = model(person, cloth)
        warped = F.grid_sample(cloth, grid, padding_mode='border', align_corners=False)

    output_path = "tryon_result.jpg"
    save_image(warped, output_path)

    st.success("✅ Try-on image generated (with dummy pose/parsing)")
    st.image(output_path, caption="Try-On Result", use_column_width=True)

    st.warning(
        "⚠️ This result uses dummy pose & segmentation. "
        "For realistic results, integrate a pose estimator and human parsing model."
    )

else:
    st.info("Please upload both images to proceed.")
