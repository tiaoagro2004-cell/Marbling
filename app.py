import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Análise de Marmoreio", layout="centered")

def processar_marmoreio(img_opencv):
    # Converter para LAB
    lab = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Máscaras
    _, mask_carne = cv2.threshold(a, 130, 255, cv2.THRESH_BINARY)
    _, mask_gordura = cv2.threshold(l, 165, 255, cv2.THRESH_BINARY)
    gordura_intra = cv2.bitwise_and(mask_gordura, mask_gordura, mask=mask_carne)
    
    pixels_carne = cv2.countNonZero(mask_carne)
    pixels_gordura = cv2.countNonZero(gordura_intra)
    
    if pixels_carne == 0:
        return 0, "Área Inválida"
        
    porc = (pixels_gordura / pixels_carne) * 100
    
    if porc < 1.5: esc = "1.0"
    elif porc < 2.5: esc = "2.0"
    elif porc < 3.5: esc = "3.0"
    elif porc < 4.5: esc = "4.0"
    elif porc < 5.5: esc = "5.0"
    else: esc = "6.0+"
    
    return porc, esc, gordura_intra

st.title("🥩 Analisador de Marmoreio")

upload = st.file_uploader("Carregue a foto do corte", type=['jpg', 'png', 'jpeg'])

if upload:
    # Ler imagem
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Processar
    p, e, mask = processar_marmoreio(img)
    
    # Mostrar resultados
    st.metric("Escore NPPC", e)
    st.write(f"Percentual de gordura: **{p:.2f}%**")
    
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    col2.image(mask, caption="Gordura Detectada")
