import streamlit as st
import cv2
import numpy as np

# Configuração da página
st.set_page_config(page_title="Analisador de Marmoreio", layout="wide")

def processar_marmoreio(imagem_cortada):
    # Converter para o espaço de cores LAB
    lab = cv2.cvtColor(imagem_cortada, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Criar uma máscara para a carne (tons avermelhados no canal 'a')
    _, mask_carne = cv2.threshold(a, 130, 255, cv2.THRESH_BINARY)

    # Criar uma máscara para a gordura (pixels claros no canal 'L')
    _, mask_gordura = cv2.threshold(l, 165, 255, cv2.THRESH_BINARY)

    # Isolar apenas a gordura que está DENTRO da área da carne
    gordura_intramuscular = cv2.bitwise_and(mask_gordura, mask_gordura, mask=mask_carne)

    # Cálculos
    pixels_carne = cv2.countNonZero(mask_carne)
    pixels_gordura = cv2.countNonZero(gordura_intramuscular)

    if pixels_carne == 0:
        return 0, "Área inválida", gordura_intramuscular

    porcentagem = (pixels_gordura / pixels_carne) * 100

    # Escala NPPC 1999
    if porcentagem < 1.5: escore = "1.0"
    elif porcentagem < 2.5: escore = "2.0"
    elif porcentagem < 3.5: escore = "3.0"
    elif porcentagem < 4.5: escore = "4.0"
    elif porcentagem < 5.5: escore = "5.0"
    else: escore = "6.0 ou superior"

    return porcentagem, escore, gordura_intramuscular

# Interface Streamlit
st.title("🥩 Analisador de Marmoreio Suíno Premium CR_Agro")
st.markdown("---")

col_input, col_res = st.columns([1, 1])

with col_input:
    st.subheader("1. Entrada de Imagem")
    arquivo = st.file_uploader("Selecione a foto do lombo", type=["jpg", "jpeg", "png"])
    
    if arquivo:
        # CORREÇÃO: np.uint8 com "np." para evitar erro de variável não definida
        file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Imagem Original", use_container_width=True)

with col_res:
    st.subheader("2. Resultado da Análise")
    if arquivo and st.button("Executar Análise"):
        # Ajuste de ROI: Pegando um quadrado central menor para evitar a gordura de cobertura (capa)
        h, w, _ = img.shape
        # Recorte de 35% a 85% para focar no miolo do músculo
        r_start, r_end = int(h * 0.35), int(h * 0.85)
        c_start, c_end = int(w * 0.35), int(w * 0.85)
        amostragem = img[r_start:r_end, c_start:c_end]

        # Processamento
        porc, nppc, mask_visual = processar_marmoreio(amostragem)

        # Exibição de Métricas
        st.metric("Escore NPPC Estimado", nppc)
        st.write(f"**Gordura Intramuscular:** {porc:.2f}%")
        
        # Comparação Visual
        st.write("Visualização da Detecção (Branco = Gordura):")
        amostragem_rgb = cv2.cvtColor(amostragem, cv2.COLOR_BGR2RGB)
        
        # Criar uma imagem lado a lado
        res_visual = np.hstack((
            cv2.resize(amostragem_rgb, (300, 300)),
            cv2.resize(cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2RGB), (300, 300))
        ))
        st.image(res_visual, use_container_width=True)
    else:
        st.info("Aguardando upload e clique no botão para analisar.")

st.markdown("---")
st.caption("Nota: Para maior precisão, garanta que a foto esteja bem iluminada e o músculo centralizado.")
