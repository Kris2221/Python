import pygame
import sys
import random

# --- 1. Configurações Iniciais e Constantes ---

pygame.init()

# Configurações da tela
LARGURA_TELA = 800
ALTURA_TELA = 600
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Pong com IA (Rede Neural Simples)")

# Cores
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
VERDE = (0, 200, 0)
CINZA_ESCURO = (50, 50, 50)
CINZA_CLARO = (150, 150, 150)


# Configurações do jogo
LARGURA_RAQUETE = 15
ALTURA_RAQUETE = 100
TAMANHO_BOLA = 15
VELOCIDADE_RAQUETE_JOGADOR = 7
LIMITE_VELOCIDADE_IA = 7
VIDAS_MAXIMAS = 5 # <--- NOVO: Total de vidas

# Configurações das barras de vida
LARGURA_BARRA_VIDA = 200
ALTURA_BARRA_VIDA = 25

# Relógio para controlar o FPS
relogio = pygame.time.Clock()

# Fontes
fonte_placar = pygame.font.Font(None, 74)
fonte_menu = pygame.font.Font(None, 40)
fonte_menu_pequena = pygame.font.Font(None, 28)

# --- Imagem de fundo ---
# Caminho: C:\Users\brazi\Desktop\Python testes\Angela\__pycache__\Imagem_projeto_angela.jpg
FUNDO_IMAGEM = None
try:
    caminho_imagem = r"C:\Users\brazi\Desktop\Python testes\Angela\__pycache__\Imagem_projeto_angela.jpg"
    _img = pygame.image.load(caminho_imagem)
    _img = pygame.transform.scale(_img, (LARGURA_TELA, ALTURA_TELA)).convert()
    FUNDO_IMAGEM = _img
except Exception as e:
    # Falha ao carregar a imagem: continuar com fundo sólido
    print(f"Aviso: não foi possível carregar a imagem de fundo ('{caminho_imagem}'): {e}")
 

# --- 2. Classes do Jogo (Sem alterações) ---

class Bola:
    """ Classe para a bola do jogo. """
    def __init__(self):
        self.rect = pygame.Rect(0, 0, TAMANHO_BOLA, TAMANHO_BOLA)
        self.reiniciar()

    def reiniciar(self):
        self.rect.center = (LARGURA_TELA // 2, ALTURA_TELA // 2)
        self.velocidade_x = 7 * random.choice((1, -1))
        self.velocidade_y = 7 * random.choice((1, -1))

    def atualizar(self):
        self.rect.x += self.velocidade_x
        self.rect.y += self.velocidade_y
        if self.rect.top <= 0 or self.rect.bottom >= ALTURA_TELA:
            self.velocidade_y *= -1

    def desenhar(self):
        pygame.draw.ellipse(tela, BRANCO, self.rect)

class Raquete:
    """ Classe para a raquete controlada pelo Jogador. """
    def __init__(self, pos_x):
        self.rect = pygame.Rect(pos_x, ALTURA_TELA // 2 - ALTURA_RAQUETE // 2, LARGURA_RAQUETE, ALTURA_RAQUETE)
        self.velocidade = VELOCIDADE_RAQUETE_JOGADOR

    def atualizar(self, teclas):
        if teclas[pygame.K_UP]:
            self.rect.y -= self.velocidade
        if teclas[pygame.K_DOWN]:
            self.rect.y += self.velocidade
        self.rect.y = max(0, min(self.rect.y, ALTURA_TELA - ALTURA_RAQUETE))

    def desenhar(self):
        pygame.draw.rect(tela, BRANCO, self.rect)

class RaqueteIA:
    """ Classe para a raquete controlada pela IA. """
    def __init__(self, pos_x, eh_raquete_esquerda=False):
        self.rect = pygame.Rect(pos_x, ALTURA_TELA // 2 - ALTURA_RAQUETE // 2, LARGURA_RAQUETE, ALTURA_RAQUETE)
        self.limite_velocidade = LIMITE_VELOCIDADE_IA
        self.eh_raquete_esquerda = eh_raquete_esquerda
        self.peso_p = 0.08
        self.peso_d = 0.04
        self.erro_anterior = 0

    def atualizar(self, bola):
        erro_atual = bola.rect.centery - self.rect.centery
        derivativo = erro_atual - self.erro_anterior
        velocidade_movimento = (self.peso_p * erro_atual) + (self.peso_d * derivativo)
        velocidade_movimento = max(-self.limite_velocidade, min(self.limite_velocidade, velocidade_movimento))

        movendo_na_minha_direcao = (self.eh_raquete_esquerda and bola.velocidade_x < 0) or \
                                   (not self.eh_raquete_esquerda and bola.velocidade_x > 0)
        if movendo_na_minha_direcao:
            self.rect.y += velocidade_movimento
        
        self.erro_anterior = erro_atual
        self.rect.y = max(0, min(self.rect.y, ALTURA_TELA - ALTURA_RAQUETE))

    def desenhar(self):
        pygame.draw.rect(tela, BRANCO, self.rect)


class Confete:
    """Partícula simples para efeito de confete/celebração quando um ponto é marcado."""
    def __init__(self, pos_x, pos_y, vel_x, vel_y, cor, tamanho, vida):
        # Posições/velocidades em float para movimento suave
        self.x = float(pos_x)
        self.y = float(pos_y)
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        self.cor = cor
        self.tamanho = int(tamanho)
        self.vida = int(vida)  # contagem de frames restantes

    def atualizar(self):
        # Aplica movimento
        self.x += self.vel_x
        self.y += self.vel_y

        # Gravidade simples (puxa para baixo)
        self.vel_y += 0.25

        # Leve atrito para desacelerar
        self.vel_x *= 0.995
        self.vel_y *= 0.995

        # Diminui vida
        self.vida -= 1

    def desenhar(self):
        # Desenha como pequeno círculo
        pygame.draw.circle(tela, self.cor, (int(self.x), int(self.y)), max(1, self.tamanho))

    def esta_viva(self):
        # Considera viva se ainda tiver vida e estiver, ao menos parcialmente, dentro de limites razoáveis
        if self.vida <= 0:
            return False
        if self.x < -50 or self.x > LARGURA_TELA + 50 or self.y < -50 or self.y > ALTURA_TELA + 50:
            return False
        return True


# --- 3. Funções Auxiliares (Modificadas) ---

def desenhar_texto(texto, fonte, cor, superficie, x, y, centro=False):
    """ Função auxiliar para desenhar texto na tela. """
    textobj = fonte.render(texto, 1, cor)
    textrect = textobj.get_rect()
    if centro:
        textrect.center = (x, y)
    else:
        textrect.topleft = (x, y)
    superficie.blit(textobj, textrect)
    # Não retorna mais o rect, pois a função de botão fará isso

def desenhar_botao(texto, fonte, x, y, largura, altura, cor_fundo, cor_borda):
    """
    NOVA FUNÇÃO: Desenha um botão clicável e retorna seu retângulo.
    O 'x' e 'y' são o *centro* do botão.
    """
    # Cria o retângulo do botão centralizado
    rect_botao = pygame.Rect(x - largura // 2, y - altura // 2, largura, altura)
    
    # Desenha o fundo do botão
    pygame.draw.rect(tela, cor_fundo, rect_botao, border_radius=10)
    
    # Desenha a borda do botão
    pygame.draw.rect(tela, cor_borda, rect_botao, 2, border_radius=10)
    
    # Desenha o texto centralizado no botão
    desenhar_texto(texto, fonte, BRANCO, tela, rect_botao.centerx, rect_botao.centery, centro=True)
    
    # Retorna o retângulo para detecção de clique
    return rect_botao

def desenhar_barra_vida(superficie, x, y, vidas_atuais):
    """
    NOVA FUNÇÃO: Desenha a barra de vida.
    O 'x' e 'y' são o canto superior esquerdo da barra.
    """
    # Calcula a porcentagem de vida
    porcentagem = max(0, vidas_atuais / VIDAS_MAXIMAS)
    largura_vida_atual = LARGURA_BARRA_VIDA * porcentagem
    
    # Retângulos da barra
    rect_fundo = pygame.Rect(x, y, LARGURA_BARRA_VIDA, ALTURA_BARRA_VIDA)
    rect_vida = pygame.Rect(x, y, largura_vida_atual, ALTURA_BARRA_VIDA)
    
    # Desenha o fundo (cinza escuro)
    pygame.draw.rect(superficie, CINZA_ESCURO, rect_fundo, border_radius=5)
    # Desenha a vida (verde)
    pygame.draw.rect(superficie, VERDE, rect_vida, border_radius=5)
    # Desenha a borda (branca)
    pygame.draw.rect(superficie, BRANCO, rect_fundo, 2, border_radius=5)


def tratar_colisao(bola, raquete1, raquete2):
    """ Lida com as colisões da bola com as raquetes. """
    if bola.rect.colliderect(raquete1.rect) and bola.velocidade_x < 0:
        bola.velocidade_x *= -1
        delta_y = bola.rect.centery - raquete1.rect.centery
        bola.velocidade_y = delta_y * 0.15
    if bola.rect.colliderect(raquete2.rect) and bola.velocidade_x > 0:
        bola.velocidade_x *= -1
        delta_y = bola.rect.centery - raquete2.rect.centery
        bola.velocidade_y = delta_y * 0.15


# --- 4. Menu Principal (Modificado para botões) ---

def menu_principal():
    """ Mostra o menu principal e aguarda a seleção do modo de jogo. """
    
    while True:
        # Desenha o fundo (imagem se disponível, senão cor sólida)
        if FUNDO_IMAGEM:
            tela.blit(FUNDO_IMAGEM, (0, 0))
        else:
            tela.fill(PRETO)
        
        desenhar_texto("PONG COM IA", fonte_placar, BRANCO, tela, LARGURA_TELA / 2, 150, centro=True)
        
        # --- Desenha os botões clicáveis ---
        # A função desenhar_botao retorna o 'rect' que precisamos para checar o clique
        rect_modo1 = desenhar_botao("Jogador vs IA", fonte_menu, LARGURA_TELA / 2, 300, 300, 60, CINZA_ESCURO, BRANCO)
        rect_modo2 = desenhar_botao("IA vs IA", fonte_menu, LARGURA_TELA / 2, 380, 300, 60, CINZA_ESCURO, BRANCO)
        
        desenhar_texto("(Clique para selecionar)", fonte_menu_pequena, CINZA_CLARO, tela, LARGURA_TELA / 2, 450, centro=True)

        pygame.display.flip()

        # --- Loop de eventos do menu ---
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # NOVO: Checa por clique do mouse
            if evento.type == pygame.MOUSEBUTTONDOWN:
                pos_mouse = pygame.mouse.get_pos()
                
                # Checa se o clique foi dentro do 'rect' do botão 1
                if rect_modo1.collidepoint(pos_mouse):
                    return "JOGADOR_vs_IA"  # Retorna o modo
                    
                # Checa se o clique foi dentro do 'rect' do botão 2
                if rect_modo2.collidepoint(pos_mouse):
                    return "IA_vs_IA" # Retorna o modo

# --- 5. Loop Principal do Jogo (Modificado) ---

def loop_jogo(modo_de_jogo):
    """ O loop principal onde o jogo acontece. """
    
    # Cria os objetos do jogo
    bola = Bola()
    
    # --- NOVO: Lógica de Vidas ---
    vidas1 = VIDAS_MAXIMAS
    vidas2 = VIDAS_MAXIMAS

    # Lista de confetes/partículas ativas
    confetes_ativos = []

    # Define a posição X central das barras de vida
    pos_x_barra1 = LARGURA_TELA / 4 - LARGURA_BARRA_VIDA / 2
    pos_x_barra2 = LARGURA_TELA * 3 / 4 - LARGURA_BARRA_VIDA / 2

    # --- NOVO: Define o retângulo do botão de voltar ANTES do loop ---
    # Isso permite que ele seja checado no loop de eventos e desenhado depois
    rect_botao_voltar = pygame.Rect(LARGURA_TELA / 2 - 100, ALTURA_TELA - 45, 200, 35)

    # Inicialização dos objetos (raquetes)
    if modo_de_jogo == "JOGADOR_vs_IA":
        raquete1 = Raquete(pos_x=30)
    else:
        raquete1 = RaqueteIA(pos_x=30, eh_raquete_esquerda=True)
        raquete1.peso_p = 0.09
        raquete1.peso_d = 0.05
    
    raquete2 = RaqueteIA(pos_x=LARGURA_TELA - 30 - LARGURA_RAQUETE, eh_raquete_esquerda=False)
    
    # --- LOOP DO JOGO ---
    rodando = True
    while rodando:
        
        # 1. Manipulação de Eventos
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # --- NOVO: Checa clique no botão de voltar ---
            if evento.type == pygame.MOUSEBUTTONDOWN:
                if rect_botao_voltar.collidepoint(evento.pos):
                    rodando = False # Quebra o loop do jogo e volta ao menu
                    # 'return' também funcionaria aqui

        teclas = pygame.key.get_pressed()

        # 2. Atualização da Lógica do Jogo
        bola.atualizar()
        
        if modo_de_jogo == "JOGADOR_vs_IA":
            raquete1.atualizar(teclas)
        else:
            raquete1.atualizar(bola)
        
        raquete2.atualizar(bola)

        # 3. Lógica de Colisão
        tratar_colisao(bola, raquete1, raquete2)

        # 4. Lógica de Pontuação (agora Vidas)
        if bola.rect.left <= 0:
            # Raquete 1 (esquerda) perde vida
            vidas1 -= 1
            # Gera confetes no local do ponto (lado esquerdo)
            px, py = bola.rect.centerx, bola.rect.centery
            # cores variadas para o confete
            CORES_CONFETE = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 200, 0), (0, 200, 200), (0, 0, 255), (255, 0, 255)]
            for _ in range(80):
                vx = random.uniform(-6, 6)
                vy = random.uniform(-8, -2)
                cor = random.choice(CORES_CONFETE)
                tam = random.randint(2, 5)
                vida = random.randint(40, 80)
                confetes_ativos.append(Confete(px, py, vx, vy, cor, tam, vida))
            bola.reiniciar()
        
        if bola.rect.right >= LARGURA_TELA:
            # Raquete 2 (direita) perde vida
            vidas2 -= 1
            # Gera confetes no local do ponto (lado direito)
            px, py = bola.rect.centerx, bola.rect.centery
            CORES_CONFETE = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 200, 0), (0, 200, 200), (0, 0, 255), (255, 0, 255)]
            for _ in range(80):
                vx = random.uniform(-6, 6)
                vy = random.uniform(-8, -2)
                cor = random.choice(CORES_CONFETE)
                tam = random.randint(2, 5)
                vida = random.randint(40, 80)
                confetes_ativos.append(Confete(px, py, vx, vy, cor, tam, vida))
            bola.reiniciar()
            
        # --- NOVO: Checagem de Fim de Jogo ---
        if vidas1 <= 0 or vidas2 <= 0:
            if FUNDO_IMAGEM:
                tela.blit(FUNDO_IMAGEM, (0, 0))
            else:
                tela.fill(PRETO)
            
            # Define o texto do vencedor
            if modo_de_jogo == "JOGADOR_vs_IA":
                vencedor_texto = "IA Venceu!" if vidas1 <= 0 else "Jogador Venceu!"
            else:
                vencedor_texto = "IA 2 Venceu!" if vidas1 <= 0 else "IA 1 Venceu!"

            desenhar_texto(vencedor_texto, fonte_placar, BRANCO, tela, LARGURA_TELA / 2, ALTURA_TELA / 2 - 40, centro=True)
            desenhar_texto("Voltando ao menu...", fonte_menu, CINZA_CLARO, tela, LARGURA_TELA / 2, ALTURA_TELA / 2 + 30, centro=True)
            
            pygame.display.flip()
            pygame.time.wait(3000) # Espera 3 segundos
            rodando = False # Termina o loop do jogo

        # 5. Desenho (Renderização)
        # Desenha o fundo (imagem se disponível, senão cor sólida)
        if FUNDO_IMAGEM:
            tela.blit(FUNDO_IMAGEM, (0, 0))
        else:
            tela.fill(PRETO)
        
        # Desenha a linha central
        pygame.draw.aaline(tela, BRANCO, (LARGURA_TELA // 2, 0), (LARGURA_TELA // 2, ALTURA_TELA))
        
        # Desenha os objetos
        bola.desenhar()
        raquete1.desenhar()
        raquete2.desenhar()
        
        
        # --- NOVO: Desenha as Barras de Vida ---
        desenhar_barra_vida(tela, pos_x_barra1, 20, vidas1)
        desenhar_barra_vida(tela, pos_x_barra2, 20, vidas2)

        # --- Atualiza e desenha confetes ativos ---
        for conf in confetes_ativos:
            conf.atualizar()
            conf.desenhar()

        # Remove confetes mortos ou fora da tela
        confetes_ativos = [c for c in confetes_ativos if c.esta_viva()]

        # --- NOVO: Desenha o botão de voltar ---
        pygame.draw.rect(tela, CINZA_ESCURO, rect_botao_voltar, border_radius=5)
        pygame.draw.rect(tela, BRANCO, rect_botao_voltar, 1, border_radius=5)
        desenhar_texto("Voltar ao Menu", fonte_menu_pequena, BRANCO, tela, rect_botao_voltar.centerx, rect_botao_voltar.centery, centro=True)


        # 6. Atualiza a Tela
        pygame.display.flip()
        
        # Controla o FPS
        relogio.tick(60)


# --- 6. Ponto de Entrada do Programa (Modificado) ---

if __name__ == "__main__":
    
    # --- NOVO: Loop principal do programa ---
    # Isso garante que, ao sair do 'loop_jogo',
    # o programa volte para o 'menu_principal'
    while True:
        modo_selecionado = menu_principal()
        loop_jogo(modo_selecionado)

    # O código abaixo só será alcançado se sairmos do while True,
    # o que não acontece neste caso (mas é uma boa prática em outros).
    pygame.quit()
    sys.exit()