import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


class TaskEditor:
    """ A simple gui to load and interact with tasks and robots in 3D. """

    axcolor = "lightgoldenrodyellow"

    def __init__(self):
        # default values
        self.delta_f = 1.0
        self.a0 = 5
        self.f0 = 3

        # setup
        self.fig, self.ax = self.setup()

        (self.l,) = self.create_plot_lines()

        self.add_sliders()
        self.add_reset_button()
        self.add_radio_boxes()

    @staticmethod
    def setup():
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.50, bottom=0.25)
        ax.margins(x=0)
        return fig, ax

    def create_plot_lines(self):
        self.t = np.arange(0.0, 1.0, 0.001)
        s = self.a0 * np.sin(2 * np.pi * self.f0 * self.t)
        return self.ax.plot(self.t, s, lw=2)

    def update_plot_lines(self, val):
        # ingnore callback value and read them from inputs manually
        amp = self.samp.val
        freq = self.sfreq.val
        self.l.set_ydata(amp * np.sin(2 * np.pi * freq * self.t))
        self.fig.canvas.draw_idle()

    def reset_sliders(self, event):
        self.sfreq.reset()
        self.samp.reset()

    def colorfunc(self, label):
        self.l.set_color(label)
        self.fig.canvas.draw_idle()

    def add_sliders(self):
        self.axfreq = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=self.axcolor)
        self.axamp = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=self.axcolor)
        self.sfreq = Slider(
            self.axfreq, "Freq", 0.1, 30.0, valinit=self.f0, valstep=self.delta_f
        )
        self.samp = Slider(self.axamp, "Amp", 0.1, 10.0, valinit=self.a0)

        self.sfreq.on_changed(lambda val: self.update_plot_lines(val))
        self.samp.on_changed(lambda val: self.update_plot_lines(val))

    def add_reset_button(self):
        self.resetax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(
            self.resetax, "Reset", color=self.axcolor, hovercolor="0.975"
        )
        self.button.on_clicked(lambda event: self.reset_sliders(event))

    def add_radio_boxes(self):
        self.rax = self.fig.add_axes([0.025, 0.5, 0.15, 0.15], facecolor=self.axcolor)
        self.radio = RadioButtons(self.rax, ("red", "blue", "green"), active=0)
        self.radio.on_clicked(lambda label: self.colorfunc(label))


gui = TaskEditor()
plt.show()
