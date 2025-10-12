# mlp_tk_slides.py
# Interface Tkinter que implementa um MLP (1-hidden) com backpropagation passo-a-passo,
# seguindo a notação didática (v_j_i, z_in_j, delta_j, etc.). Mostra tudo na janela,
# incluindo sweep/varredura de parâmetros (botão extra).
#
# Requisitos: numpy, matplotlib
# Execute: python mlp_tk_slides.py

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import math
import threading
import time

# Matplotlib embedding
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -------------------------
# Funções de ativação
# -------------------------
def tanh(x): 
    return np.tanh(x)

def tanh_derivative_from_output(y):
    # se y = tanh(x) -> derivative = 1 - y^2
    return 1.0 - y**2

# -------------------------
# MLP passo-a-passo (didático)
# -------------------------
class MLP_Slides:
    def __init__(self, n_inputs=1, n_hidden=6, n_outputs=1, init_range=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # v_j_i: pesos entrada->oculto (inclui bias i=0)
        self.v = np.random.uniform(-init_range, init_range, (n_hidden, n_inputs + 1))
        # w_k_j: pesos oculto->saida (inclui bias j=0)
        self.w = np.random.uniform(-init_range, init_range, (n_outputs, n_hidden + 1))
        self.prev_delta_v = np.zeros_like(self.v)
        self.prev_delta_w = np.zeros_like(self.w)
        self._cache = {}

    def forward(self, x):
        # x sem bias: array shape (n_inputs,)
        # 1) x_with_bias: x0 = 1
        x_with_bias = np.concatenate(([1.0], x))  # shape (n_inputs+1,)
        # 2) z_in_j = sum_i v_j_i * x_with_bias_i
        z_in = self.v.dot(x_with_bias)            # shape (n_hidden,)
        # 3) z_j = tanh(z_in_j)
        z = tanh(z_in)
        # 4) z_with_bias (z0 = 1)
        z_with_bias = np.concatenate(([1.0], z))
        # 5) y_in_k = sum_j w_k_j * z_with_bias_j
        y_in = self.w.dot(z_with_bias)            # shape (n_outputs,)
        # 6) y_k = tanh(y_in_k)
        y = tanh(y_in)
        # cache for backprop
        self._cache = {
            'x_with_bias': x_with_bias,
            'z_in': z_in,
            'z': z,
            'z_with_bias': z_with_bias,
            'y_in': y_in,
            'y': y
        }
        return y.copy()

    def train_pattern(self, x, t, lr=0.01, momentum=0.0, verbose=False):
        """
        Treina usando um padrão (atualização online).
        Retorna erro instantâneo 0.5 * sum(e_k^2).
        Se verbose=True imprime passo a passo (útil para checagem com slides).
        """
        # forward
        y = self.forward(x)  # shape (n_outputs,)
        # erro
        e = t - y  # shape (n_outputs,)
        # delta de saída: delta_k = e_k * (1 - y_k^2)
        delta_out = e * tanh_derivative_from_output(y)  # shape (n_outputs,)
        # contribuição para hidden: delta_in_hidden_j = sum_k w_k_j * delta_out_k
        # skip bias w[:,0]
        delta_in_hidden = self.w[:, 1:].T.dot(delta_out)  # shape (n_hidden,)
        # delta hidden: delta_j = delta_in_hidden_j * (1 - z_j^2)
        z = self._cache['z']
        delta_hidden = delta_in_hidden * tanh_derivative_from_output(z)  # shape (n_hidden,)
        # updates
        z_with_bias = self._cache['z_with_bias']
        delta_w = np.outer(delta_out, z_with_bias) * lr
        x_with_bias = self._cache['x_with_bias']
        delta_v = np.outer(delta_hidden, x_with_bias) * lr
        # momentum
        self.prev_delta_w = momentum * self.prev_delta_w + delta_w
        self.prev_delta_v = momentum * self.prev_delta_v + delta_v
        # update weights
        self.w += self.prev_delta_w
        self.v += self.prev_delta_v

        if verbose:
            # imprime passo-a-passo (sintetizado)
            print("----- STEP (verbose) -----")
            print("x:", x)
            print("x_with_bias:", np.round(x_with_bias, 6))
            print("z_in:", np.round(self._cache['z_in'], 6))
            print("z:", np.round(z, 6))
            print("z_with_bias:", np.round(z_with_bias, 6))
            print("y_in:", np.round(self._cache['y_in'], 6))
            print("y:", np.round(y, 6))
            print("t:", t)
            print("e:", np.round(e, 6))
            print("delta_out:", np.round(delta_out, 6))
            print("delta_hidden:", np.round(delta_hidden, 6))
            print("delta_w (lr*):", np.round(delta_w, 6))
            print("delta_v (lr*):", np.round(delta_v, 6))
            print("-------------------------")

        return 0.5 * np.sum(e**2)

    def predict(self, X):
        # X: array shape (n_samples, n_inputs)
        out = []
        for x in X:
            y = self.forward(x)
            out.append(y[0])
        return np.array(out)


# -------------------------
# Geração de dados f(x) = sin(x)*sin(2x)
# -------------------------
def generate_data(n_samples=100, domain=(-math.pi, math.pi), seed=None):
    if seed is not None:
        np.random.seed(seed)
    xs = np.linspace(domain[0], domain[1], n_samples)
    ys = np.sin(xs) * np.sin(2 * xs)
    X = xs.reshape(-1, 1)
    Y = ys.reshape(-1)
    return X, Y


# -------------------------
# Treinamento (estrutura de slides 47/48)
# -------------------------
def train_network(model, X_train, Y_train, lr=0.05, momentum=0.8, epochs=200, verbose_first_n=0, text_callback=None, progress_callback=None):
    """
    Treina a rede:
    - model: instância MLP_Slides
    - X_train, Y_train: dados
    - lr, momentum, epochs
    - verbose_first_n: quantos padrões da 1ª época mostrar passo-a-passo
    - text_callback: função(text:str) -> mostrar logs na GUI (opcional)
    - progress_callback: função(progress: float) -> para atualizar barra de progresso (0..1) (opcional)
    Retorna: mse_history (list), final_mse
    """
    N = len(X_train)
    mse_history = []
    for epoch in range(epochs):
        sq_err_sum = 0.0
        perm = np.random.permutation(N)
        for i, idx in enumerate(perm):
            x = X_train[idx]
            t = np.array([Y_train[idx]])
            verbose = (epoch == 0 and i < verbose_first_n)
            err = model.train_pattern(x, t, lr=lr, momentum=momentum, verbose=verbose)
            sq_err_sum += err
        mse = (2.0 * sq_err_sum) / N  # conforme notação usada antes
        mse_history.append(mse)
        if text_callback:
            text_callback(f"Epoch {epoch+1}/{epochs}  MSE={mse:.6f}")
        if progress_callback:
            progress_callback((epoch+1)/epochs)
    final_mse = mse_history[-1]
    return mse_history, final_mse


# -------------------------
# GUI
# -------------------------
class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("MLP Backprop (slides 47-48) - Interface")
        self.model = None
        self.mse_history = []
        self.X = None
        self.Y = None
        self.pred = None

        # Layout
        main = ttk.Frame(root, padding=8)
        main.grid(sticky='nsew')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # PARAMS frame
        frm_params = ttk.LabelFrame(main, text="Parâmetros", padding=8)
        frm_params.grid(row=0, column=0, sticky='nw')

        # entries
        self.var_lr = tk.StringVar(value="0.05")
        self.var_mom = tk.StringVar(value="0.8")
        self.var_epochs = tk.StringVar(value="200")
        self.var_hidden = tk.StringVar(value="6")
        self.var_init_range = tk.StringVar(value="0.5")
        self.var_samples = tk.StringVar(value="100")
        self.var_seed = tk.StringVar(value="42")
        self.var_verbose_count = tk.StringVar(value="2")

        params = [
            ("Taxa de aprendizado (lr)", self.var_lr),
            ("Momentum", self.var_mom),
            ("Épocas", self.var_epochs),
            ("Neurônios ocultos", self.var_hidden),
            ("Init range (|w|<=)", self.var_init_range),
            ("Amostras (treino)", self.var_samples),
            ("Seed (int)", self.var_seed),
            ("Padrões verbose (1ª época)", self.var_verbose_count)
        ]
        for i, (lbl, var) in enumerate(params):
            ttk.Label(frm_params, text=lbl).grid(row=i, column=0, sticky='w', padx=2, pady=2)
            ttk.Entry(frm_params, textvariable=var, width=10).grid(row=i, column=1, sticky='w', padx=2, pady=2)

        # Buttons frame
        frm_buttons = ttk.Frame(main)
        frm_buttons.grid(row=1, column=0, sticky='w', pady=6)
        btn_train = ttk.Button(frm_buttons, text="Treinar Rede", command=self.on_train)
        btn_train.grid(row=0, column=0, padx=6)
        btn_weights = ttk.Button(frm_buttons, text="Mostrar Pesos Finais", command=self.on_show_weights)
        btn_weights.grid(row=0, column=1, padx=6)
        btn_graphs = ttk.Button(frm_buttons, text="Mostrar Gráficos", command=self.on_show_graphs)
        btn_graphs.grid(row=0, column=2, padx=6)
        btn_sweep = ttk.Button(frm_buttons, text="Testar Parâmetros (varredura)", command=self.on_sweep)
        btn_sweep.grid(row=0, column=3, padx=6)
        btn_clear = ttk.Button(frm_buttons, text="Limpar Logs", command=self.on_clear_logs)
        btn_clear.grid(row=0, column=4, padx=6)
        btn_exit = ttk.Button(frm_buttons, text="Sair", command=root.destroy)
        btn_exit.grid(row=0, column=5, padx=6)

        # Progress bar
        self.progress = ttk.Progressbar(main, orient='horizontal', length=400, mode='determinate')
        self.progress.grid(row=2, column=0, pady=6, sticky='w')

        # Logs / outputs
        frm_output = ttk.LabelFrame(main, text="Resultados / Logs", padding=6)
        frm_output.grid(row=3, column=0, sticky='nsew')
        frm_output.columnconfigure(0, weight=1)
        frm_output.rowconfigure(0, weight=1)

        self.txt = scrolledtext.ScrolledText(frm_output, width=80, height=18, wrap='word')
        self.txt.grid(row=0, column=0, sticky='nsew')

        # Canvas for plots
        frm_plots = ttk.LabelFrame(main, text="Gráficos", padding=6)
        frm_plots.grid(row=0, column=1, rowspan=4, sticky='nsew', padx=8)
        frm_plots.columnconfigure(0, weight=1)
        frm_plots.rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(6,4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # fit
        self.ax2 = self.fig.add_subplot(212)  # mse
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm_plots)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Initialize data
        self.append_text("Pronto. Configure parâmetros e clique em 'Treinar Rede'.")

    # UTIL
    def append_text(self, msg):
        now = time.strftime("%H:%M:%S")
        self.txt.insert('end', f"[{now}] {msg}\n")
        self.txt.see('end')
        self.root.update()

    def set_progress(self, fraction):
        self.progress['value'] = fraction * 100
        self.root.update()

    # BUTTON HANDLERS
    def on_clear_logs(self):
        self.txt.delete('1.0', 'end')

    def on_train(self):
        # ler parametros
        try:
            lr = float(self.var_lr.get())
            momentum = float(self.var_mom.get())
            epochs = int(self.var_epochs.get())
            n_hidden = int(self.var_hidden.get())
            init_range = float(self.var_init_range.get())
            n_samples = int(self.var_samples.get())
            seed = int(self.var_seed.get())
            verbose_n = int(self.var_verbose_count.get())
        except Exception as e:
            messagebox.showerror("Parâmetros inválidos", str(e))
            return

        # gerar dados
        self.X, self.Y = generate_data(n_samples=n_samples, seed=seed)
        # criar modelo
        self.model = MLP_Slides(n_inputs=1, n_hidden=n_hidden, n_outputs=1, init_range=init_range, seed=seed)
        self.append_text(f"Modelo inicializado: n_hidden={n_hidden}, init_range={init_range}, seed={seed}")
        self.append_text("Pesos iniciais (v, w) mostrados parcialmente no log.")

        # imprime alguns pesos iniciais
        self.append_text("v (input->hidden) (first rows):")
        for row in self.model.v[:min(6, len(self.model.v))]:
            self.append_text("  " + np.array2string(row, precision=6, separator=', '))
        self.append_text("w (hidden->output):")
        for row in self.model.w[:min(6, len(self.model.w))]:
            self.append_text("  " + np.array2string(row, precision=6, separator=', '))

        # bloqueia botões e treina (pode demorar)
        self.append_text("Iniciando treinamento...")
        # training in another thread so UI remains responsive
        def _train():
            self.progress['value'] = 0
            self.mse_history, final_mse = train_network(self.model, self.X, self.Y,
                                                       lr=lr, momentum=momentum, epochs=epochs,
                                                       verbose_first_n=verbose_n,
                                                       text_callback=self.append_text,
                                                       progress_callback=self.set_progress)
            # predições
            self.pred = self.model.predict(self.X)
            self.append_text(f"Treinamento concluído. MSE final: {final_mse:.6f}")
            self.plot_results()
            self.set_progress(1.0)
        t = threading.Thread(target=_train, daemon=True)
        t.start()

    def on_show_weights(self):
        if self.model is None:
            messagebox.showinfo("Info", "Treine a rede primeiro.")
            return
        self.append_text("Pesos finais v (input->hidden):")
        for row in self.model.v:
            self.append_text("  " + np.array2string(row, precision=8, separator=', '))
        self.append_text("Pesos finais w (hidden->output):")
        for row in self.model.w:
            self.append_text("  " + np.array2string(row, precision=8, separator=', '))

    def plot_results(self):
        # limpar e plotar
        if self.X is None or self.pred is None or self.mse_history is None:
            return
        xs = self.X.reshape(-1)
        # plot fit
        self.ax1.clear()
        self.ax1.set_title("f(x)=sin(x)*sin(2x)  - Rede MLP aprox.")
        self.ax1.plot(xs, self.Y, label='Alvo')
        self.ax1.plot(xs, self.pred, linestyle='--', label='MLP')
        self.ax1.legend()
        # plot mse
        self.ax2.clear()
        self.ax2.set_title("MSE por época")
        self.ax2.plot(range(1, len(self.mse_history)+1), self.mse_history)
        self.ax2.set_xlabel("Época")
        self.ax2.set_ylabel("MSE")
        self.canvas.draw()

    def on_show_graphs(self):
        if self.mse_history is None or len(self.mse_history) == 0:
            messagebox.showinfo("Info", "Treine a rede primeiro para ver gráficos.")
            return
        self.plot_results()

    def on_sweep(self):
        # pequenos conjuntos de parâmetros para varredura rápida
        try:
            n_samples = int(self.var_samples.get())
            seed = int(self.var_seed.get())
            n_hidden_base = int(self.var_hidden.get())
        except Exception as e:
            messagebox.showerror("Parâmetros inválidos", str(e))
            return

        # grade (pequena para não demorar)
        lrs = [0.01, 0.03, 0.05]
        moms = [0.0, 0.5, 0.8]
        hiddens = [max(1, n_hidden_base-2), n_hidden_base, n_hidden_base+2]
        init_ranges = [0.2, 0.5]

        total = len(lrs)*len(moms)*len(hiddens)*len(init_ranges)
        self.append_text(f"Iniciando varredura rápida ({total} combos)...")
        self.progress['value'] = 0

        # training in thread
        def _sweep():
            best = None
            best_params = None
            combo_idx = 0
            combos = []
            for lr in lrs:
                for mom in moms:
                    for nh in hiddens:
                        for ir in init_ranges:
                            combos.append((lr, mom, nh, ir))
            for idx, (lr, mom, nh, ir) in enumerate(combos):
                combo_idx += 1
                frac = combo_idx/len(combos)
                self.set_progress(frac)
                self.append_text(f"Testando combo {combo_idx}/{len(combos)}: lr={lr}, mom={mom}, nh={nh}, init_range={ir}")
                X, Y = generate_data(n_samples=n_samples, seed=seed)
                model = MLP_Slides(n_inputs=1, n_hidden=nh, n_outputs=1, init_range=ir, seed=seed)
                # treino curto para avaliar
                mse_history, final_mse = train_network(model, X, Y, lr=lr, momentum=mom, epochs=120, verbose_first_n=0)
                self.append_text(f" -> MSE(final)={final_mse:.6f}")
                if best is None or final_mse < best:
                    best = final_mse
                    best_params = (lr, mom, nh, ir, model, mse_history)
            # fim combos
            self.append_text("Varredura concluída.")
            self.append_text(f"Melhor MSE: {best:.6f} com lr={best_params[0]}, mom={best_params[1]}, nh={best_params[2]}, init_range={best_params[3]}")
            # define modelo e resultados para visualização
            self.model = best_params[4]
            self.X, self.Y = generate_data(n_samples=n_samples, seed=seed)
            self.pred = self.model.predict(self.X)
            self.mse_history = best_params[5]
            self.plot_results()
            self.set_progress(1.0)

        t = threading.Thread(target=_sweep, daemon=True)
        t.start()


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()
