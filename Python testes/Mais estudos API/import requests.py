import requests
import base64
import os
import json

# Função para carregar a imagem e convertê-la para Base64
def carregar_imagem_base64(caminho_imagem):
    with open(caminho_imagem, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    return image_base64

# Função para enviar a imagem em Base64 e rotacioná-la
def rotacionar_imagem(image_base64, grau):
    url = "https://alunos.umg.com.br/webhook/rotacionar"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "e6c2684cfec66aa67b6c"  # substitua por sua chave válida
    }

    data = {
        "texto_base64": image_base64,
        "grau": grau
    }

    response = requests.post(url, json=data, headers=headers)

    print("✅ Requisição enviada.")
    print("📦 Status:", response.status_code)

    try:
        # Tenta interpretar a resposta como JSON
        response_data = response.json()
        print("📄 JSON retornado pela API:")
        print(json.dumps(response_data, indent=4))

        for chave in response_data.keys():
            print(f"🔍 Campo encontrado: {chave}")

        # Novo tratamento para o campo 'resposta'
        conteudo_resposta = response_data.get("resposta")
        print("\n📨 Conteúdo do campo 'resposta':")
        print(conteudo_resposta)

        # Se for uma string base64 grande, tenta salvar como imagem
        if isinstance(conteudo_resposta, str) and len(conteudo_resposta) > 100:
            try:
                salvar_imagem_base64(conteudo_resposta, "imagem_rotacionada.jpg")
            except Exception as e:
                print(f"❌ Não foi possível salvar a imagem: {e}")
        else:
            print("❌ O campo 'resposta' não parece conter uma imagem em base64.")

    except ValueError as e:
        print("❌ Erro ao converter resposta em JSON.")
        print("🧾 Conteúdo bruto da resposta:")
        print(response.text)

# Função para salvar a imagem rotacionada localmente
def salvar_imagem_base64(image_base64, nome_arquivo):
    pasta_destino = os.path.join(os.path.expanduser("~"), "Downloads")
    caminho_saida = os.path.join(pasta_destino, nome_arquivo)

    with open(caminho_saida, "wb") as img_file:
        img_file.write(base64.b64decode(image_base64))

    print(f"📁 Imagem salva como: {caminho_saida}")

# Caminho da imagem original - atualize se necessário
caminho_da_imagem = r"C:\Users\brazi\Downloads\krishna.jpg"

# Executa o processo
imagem_base64 = carregar_imagem_base64(caminho_da_imagem)
rotacionar_imagem(imagem_base64, 45)




