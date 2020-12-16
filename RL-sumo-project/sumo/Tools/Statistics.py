import numpy as np
# import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt


class Observer:
    def __init__(self):
        self.plot_num = 0
        self.dyn_plot_var_buffer = {}
        self.statistics_buffer = {}
        self.stop_and_go_pos_buffer = []
        self.stop_and_go_vel_buffer = []

    """
    Plot a list of variables dynamically
        :param var: list of vars which will be plot; multiple vars can be passed by the form of list
        :param step: current step in running env, which will be the reference of X axis
        :param plot_term: the interval between two adjacent plots
        :param plot_range: only plot latest data while the latest range will be determined by this param,
                           and old data will be removed from graph; if you have multiple vars, you can 
                           provide different range by the form of list
    """
    def plot_var_dyn(self, var: list, step, plot_term, plot_range: list, plot_num: int, color=None, subplot=True):
        if color is None:
            color = []
        assert isinstance(var, list), TypeError
        plt.figure(plot_num)
        n_sub = len(var)
        if step % plot_term == 0:
            plt.cla()
        for idx in range(len(var)):
            if 'var%d' % idx not in self.dyn_plot_var_buffer:
                self.dyn_plot_var_buffer['var%d' % idx] = []
            self.dyn_plot_var_buffer['var%d' % idx].append(var[idx])
            if plot_range[idx] and len(self.dyn_plot_var_buffer['var%d' % idx]) > plot_range[idx]:
                self.dyn_plot_var_buffer['var%d' % idx].pop(0)
            x = np.linspace(0,
                            len(self.dyn_plot_var_buffer['var%d' % idx]),
                            len(self.dyn_plot_var_buffer['var%d' % idx]))
            if step % plot_term == 0:
                if subplot:
                    plt.subplot(n_sub, 1, idx + 1)
                else:
                    plt.figure(idx + 1)
                if color:
                    plt.plot(x, self.dyn_plot_var_buffer['var%d' % idx], color[idx])
                else:
                    plt.plot(x, self.dyn_plot_var_buffer['var%d' % idx])
        if step % plot_term == 0:
            plt.pause(0.01)

    def plot_var(self, var, step, plot_term, color=None, subplot=True):
        if color is None:
            color = []
        assert isinstance(var, list), TypeError
        n_sub = len(var)
        for idx in range(len(var)):
            if 'var%d' % idx not in self.dyn_plot_var_buffer:
                self.dyn_plot_var_buffer['var%d' % idx] = []
            self.dyn_plot_var_buffer['var%d' % idx].append(var[idx])
            x = np.linspace(0, len(self.dyn_plot_var_buffer['var%d' % idx]),
                            len(self.dyn_plot_var_buffer['var%d' % idx]))
            if step % plot_term == 0:
                if subplot:
                    plt.subplot(n_sub, 1, idx + 1)
                else:
                    plt.figure()
                if color:
                    plt.plot(x, self.dyn_plot_var_buffer['var%d' % idx], color[idx])
                else:
                    plt.plot(x, self.dyn_plot_var_buffer['var%d' % idx])
        plt.show()

    @staticmethod
    def plot_csv(csv_dir, layout='merge'):
        data_set = pd.read_csv(csv_dir).to_numpy().T
        n_sub = data_set.shape[0]
        if layout == 'merge':
            plt.figure()
        for idx in range(data_set.shape[0]):
            if layout == 'subplot':
                plt.subplot(n_sub, 1, idx + 1)
                plt.plot(np.linspace(0, data_set[idx].size, data_set[idx].size), data_set[idx])
            elif layout == 'merge':
                plt.plot(np.linspace(0, data_set[idx].size, data_set[idx].size), data_set[idx])
            elif layout == 'respective':
                plt.figure()
                plt.plot(np.linspace(0, data_set[idx].size, data_set[idx].size), data_set[idx])
            else:
                raise TypeError('Layout parameter cannot be parsed! \n`merge`, `subplot`, `respective` can be used!]')

    def plot_stop_and_go_wave(self, n_car, position, vel, step, start_step, stop_step, x_lim):
        assert len(position) == n_car and len(vel) == n_car, ValueError
        if start_step <= step <= stop_step:
            plt.figure(2)
            buf_max = x_lim
            if len(position) != len(self.stop_and_go_pos_buffer) or len(vel) != len(self.stop_and_go_vel_buffer):
                self.stop_and_go_pos_buffer = []
                self.stop_and_go_vel_buffer = []
                for idx in range(len(position)):
                    self.stop_and_go_pos_buffer.append([])
                    self.stop_and_go_vel_buffer.append([])
            for idx in range(len(position)):
                if len(self.stop_and_go_pos_buffer[idx]) == buf_max or len(self.stop_and_go_vel_buffer[idx]) == buf_max:
                    self.stop_and_go_pos_buffer[idx].pop(0)
                    self.stop_and_go_vel_buffer[idx].pop(0)
                self.stop_and_go_pos_buffer[idx].append(position[idx])
                self.stop_and_go_vel_buffer[idx].append(vel[idx])
            if step % 50 == 0:
                plt.cla()
                x = np.linspace(0, len(self.stop_and_go_pos_buffer[0]) - 1, len(self.stop_and_go_pos_buffer[0]))
                for item_pos, item_vel in zip(self.stop_and_go_pos_buffer, self.stop_and_go_vel_buffer):
                    plt.scatter(x, item_pos, s=1, c=item_vel, vmin=0, vmax=6)
                    plt.xlim(max(x[-1] - buf_max, 0), x[-1])
                plt.pause(0.01)

    def statistics(self, var, var_name, step, start_step, stop_step, mode,):
        assert isinstance(var, list), TypeError
        assert isinstance(var_name, list), TypeError
        if start_step <= step <= stop_step:
            for idx in range(len(var)):
                if var_name[idx] not in self.statistics_buffer:
                    self.statistics_buffer[var_name[idx]] = []
                self.statistics_buffer[var_name[idx]].append(var[idx])
        elif step > stop_step:
            print('-' * 40, '\n', '*' * 40)
            print('- Statistics: ')
            for idx in range(len(var)):
                res = 'ERROR'
                if mode[idx] == 'std':
                    res = np.std(self.statistics_buffer[var_name[idx]])
                elif mode[idx] == 'mean':
                    res = np.mean(self.statistics_buffer[var_name[idx]])
                print('- \t%s %s: %s' % (var_name[idx], mode[idx], res))
            print('*' * 40, '\n', '-' * 40)


class Logger:
    def __init__(self):
        pass

    def save_pk(self):
        pass

    @staticmethod
    def save_csv(log_data: list or dict, data_name: str, csv_dir: str):
        assert isinstance(log_data, list) or isinstance(log_data, dict), 'Can only parse `list` or `dict` data!'
        assert isinstance(data_name, str), 'Can only parse `str` data-name!'
        dict_ = {}
        if isinstance(log_data, list):
            dict_[data_name] = log_data
        elif isinstance(log_data, dict):
            dict_ = log_data
        df = pd.DataFrame(dict_)
        df.to_csv(csv_dir, index=False, sep=',')
