from analysis import FibreImage, load
import glob
import numpy as np
import matplotlib.pyplot as plt
import os


def process_folders(path, sMin=1, sMax=50, half_step=False,
                    idxR=slice(None), idxC=slice(None), eq=False,
                    i_frac=0.9, o_rad=2, ext="tif", p_count=0.0, p_length=0.0):
    folders = [os.path.join(path, name)
               for name in os.listdir(path)
               if os.path.isdir(os.path.join(path, name))]
    for f in folders:
        process_folder(f, sMin=sMin, sMax=sMax, half_step=half_step, idxR=idxR, idxC=idxC, eq=eq,
                       i_frac=i_frac, o_rad=o_rad, p_count=p_count, p_length=p_length)


def process_folder(path, sMin=1, sMax=50, half_step=False,
                   idxR=slice(None), idxC=slice(None), eq=False,
                   i_frac=0.9, o_rad=2, ext="tif", p_count=0.0, p_length=0.0):
    files = glob.glob(path + "/*." + ext)
    a, s, c = process_files(files, sMin=sMin, sMax=sMax, half_step=half_step, idxR=idxR, idxC=idxC, eq=eq,
                            i_frac=i_frac, o_rad=o_rad, p_count=p_count, p_length=p_length)
    plot_and_save(path + "/", a, s, c)


def process_file(path, sMin=1, sMax=50, half_step=False,
                 idxR=slice(None), idxC=slice(None), eq=False,
                 i_frac=0.9, o_rad=2, p_count=0.0, p_length=0.0):
    files = [path]
    a, s, c = process_files(files, sMin=sMin, sMax=sMax, half_step=half_step, idxR=idxR, idxC=idxC, eq=eq,
                            i_frac=i_frac, o_rad=o_rad, p_count=p_count, p_length=p_length)
    plot_and_save(path + "_out_", a, s, c)


def process_files(files, sMin=1, sMax=50, half_step=False,
                  idxR=slice(None), idxC=slice(None), eq=False,
                  i_frac=0.9, o_rad=2, p_count=0.0, p_length=0.0):
    sigmas = np.arange(sMin, sMax, 0.5 if half_step else 1.0)
    diam_conv = 2.0 * (p_length / p_count)
    diams = list()
    diam_count = np.zeros(len(sigmas))
    angles = list()
    for file in files:
        im = load(file)
        im = im[idxR, idxC]
        ti = FibreImage(im, sigmas, equalize=eq)
        #_, d_whole = ti.get_scores()
        ti.set_skeleton(i_frac, o_rad)
        ti.set_scores()
        _, d_skeleton = ti.get_scores()
        diams.extend(ti.get_diameters())
        angles.extend(ti.get_orientations())
        n_sigma = np.array(d_skeleton['dominance'])
        diam_count += n_sigma
    angles = np.array(angles) * 180.0 / np.pi
    diams = np.array(diams) * diam_conv
    sigmas *= diam_conv
    return angles, sigmas, diam_count


def plot_and_save(path, angles, diams, count):
    a = np.array([diams, count])
    np.savetxt(path + "diameters.txt", a.T)
    a = np.array(angles)
    np.savetxt(path + "angles.txt", a.T)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(diams, count)
    ax[1].hist(angles, bins=50)
    #ax[2].scatter(angles, diams)
    fig.show()
