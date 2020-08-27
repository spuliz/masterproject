import os
import glob
import shutil
# from creator import create
from flask import Flask, send_from_directory, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from parser import parse_arguments

from functions.transformer import get_transforms
from functions.makeDataset import make_dataset
from functions.loadNetwork import load_network
from functions.getInput import get_input
from functions.normalize import normalize_lab, normalize_rgb, normalize_seg, denormalize_lab, denormalize_rgb
from functions.vis import vis_image, vis_patch
from functions.getInputv import get_inputv
from classes.imageFolder import ImageFolder
from classes.textureGan import TextureGAN
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


SKETCH_FOLDER = '/Users/spuliz/Desktop/schedio/img/val_skg/wendy'
VAL_SEG_FOLDER = '/Users/spuliz/Desktop/schedio/img/val_seg/wendy'
VAL_ERODED_FOLDER = '/Users/spuliz/Desktop/schedio/img/eroded_val_seg/wendy'
IMG_FOLDER = '/Users/spuliz/Desktop/schedio/img/val_img/wendy'
TEXTURE_FOLDER = '/Users/spuliz/Desktop/schedio/img/val_txt/wendy'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}




app = Flask(__name__, static_url_path='/Users/spuliz/Desktop/schedio/static')

# The absolute path of the directory containing images for users to upload
app.config['SKETCH_FOLDER'] = SKETCH_FOLDER
app.config['VAL_SEG_FOLDER'] = VAL_SEG_FOLDER
app.config['VAL_ERODED_FOLDER'] = VAL_ERODED_FOLDER
app.config['TEXTURE_FOLDER'] = TEXTURE_FOLDER
# The absolute path of the directory containing images for users to download
app.config["CLIENT_IMAGES"] = "/Users/spuliz/Desktop/schedio/img/output"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['SKETCH_FOLDER'], 'sketch.jpg'))
            for jpgfile in glob.iglob(os.path.join(SKETCH_FOLDER, "*.jpg")):
                shutil.copy(jpgfile, VAL_SEG_FOLDER)
                shutil.copy(jpgfile, VAL_ERODED_FOLDER)
                shutil.copy(jpgfile, IMG_FOLDER)
            # file.save(os.path.join(app.config['VAL_SEG_FOLDER'], 'sketch.jpg'))
            # file.save(os.path.join(app.config['VAL_ERODED_FOLDER'], 'sketch.jpg'))
            # return redirect(url_for('uploaded_file',filename=filename))
            return redirect(url_for('upload_texture'))
    return render_template('main.html')

@app.route('/garment', methods=['GET', 'POST'])
def upload_garment():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['SKETCH_FOLDER'], 'sketch.jpg'))
            for jpgfile in glob.iglob(os.path.join(SKETCH_FOLDER, "*.jpg")):
                shutil.copy(jpgfile, VAL_SEG_FOLDER)
                shutil.copy(jpgfile, VAL_ERODED_FOLDER)
                shutil.copy(jpgfile, IMG_FOLDER)
            # file.save(os.path.join(app.config['VAL_SEG_FOLDER'], 'sketch.jpg'))
            # file.save(os.path.join(app.config['VAL_ERODED_FOLDER'], 'sketch.jpg'))
            # return redirect(url_for('uploaded_file',filename=filename))
            return redirect(url_for('upload_cloth_texture'))
    return render_template('garment.html')



@app.route('/upload_texture', methods=['GET', 'POST'])
def upload_texture():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['TEXTURE_FOLDER'], 'texture.jpg'))
            # return redirect(url_for('uploaded_file',filename=filename))
            return redirect(url_for('profit'))
    return render_template('bag_texture.html')


@app.route('/upload_cloth_texture', methods=['GET', 'POST'])
def upload_cloth_texture():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['TEXTURE_FOLDER'], 'texture.jpg'))
            # return redirect(url_for('uploaded_file',filename=filename))
            return redirect(url_for('profit_cloth'))
    return render_template('cloth_texture.html')


@app.route('/profit', methods=['GET', 'POST'])
def profit():
    if request.method == 'POST':
        command = '--model texturegan --local_texture_size 50 --color_space lab'
        args = parse_arguments(command.split())
        args.batch_size = 1
        args.image_size =152
        args.resize_max = 256
        args.resize_min = 256
        args.data_path = '/Users/spuliz/Desktop/schedio/img' #change to your data path
        transform = get_transforms(args)
        val = make_dataset(args.data_path, 'val')
        valDset = ImageFolder('val', args.data_path, transform)
        val_display_size = 1
        valLoader = DataLoader(dataset=valDset, batch_size=val_display_size, shuffle=False)
        # pre-trained model for handbags
        model_location = '/Users/spuliz/Desktop/schedio/textureD_final_allloss_handbag_3300.pth' #change to your location
        netG = TextureGAN(5, 3, 32)
        load_network(netG, model_location)
        netG.eval()
        data = valLoader.__iter__().__next__()
        color_space = 'lab'
        img, skg, seg, eroded_seg, txt = data
        img = normalize_lab(img)
        skg = normalize_lab(skg)
        txt = normalize_lab(txt)
        seg = normalize_seg(seg)
        eroded_seg = normalize_seg(eroded_seg)
        inp,texture_loc = get_input(data,-1,-1,30,1)
        seg = seg!=0
        model = netG
        device = torch.device("cpu")
        inpv = get_inputv(inp.to(device))
        output = model(inpv.to(device))
        out_img = vis_image(denormalize_lab(output.data.double().cpu()), color_space)
        plt.figure()
        plt.imshow(np.transpose(out_img[0],(1, 2, 0)))
        plt.axis('off')
        plt.savefig('/Users/spuliz/Desktop/schedio/img/output/output.png', dpi=1000)
        return send_from_directory(app.config["CLIENT_IMAGES"], filename='output.png', as_attachment=True)
        return redirect(url_for('http://127.0.0.1:5000/profit/output.png'))
    return render_template('profit_bag.html')
    


@app.route('/profit_cloth', methods=['GET', 'POST'])
def profit_cloth():
    if request.method == 'POST':
        command = '--model texturegan --local_texture_size 50 --color_space lab'
        args = parse_arguments(command.split())
        args.batch_size = 1
        args.image_size =152
        args.resize_max = 256
        args.resize_min = 256
        args.data_path = '/Users/spuliz/Desktop/schedio/img' #change to your data path
        transform = get_transforms(args)
        val = make_dataset(args.data_path, 'val')
        valDset = ImageFolder('val', args.data_path, transform)
        val_display_size = 1
        valLoader = DataLoader(dataset=valDset, batch_size=val_display_size, shuffle=False)
        # pre-trained model for handbags
        model_location = '/Users/spuliz/Desktop/schedio/final_cloth_finetune.pth' #change to your location
        netG = TextureGAN(5, 3, 32)
        load_network(netG, model_location)
        netG.eval()
        data = valLoader.__iter__().__next__()
        color_space = 'lab'
        img, skg, seg, eroded_seg, txt = data
        img = normalize_lab(img)
        skg = normalize_lab(skg)
        txt = normalize_lab(txt)
        seg = normalize_seg(seg)
        eroded_seg = normalize_seg(eroded_seg)
        inp,texture_loc = get_input(data,-1,-1,30,1)
        seg = seg!=0
        model = netG
        device = torch.device("cpu")
        inpv = get_inputv(inp.to(device))
        output = model(inpv.to(device))
        out_img = vis_image(denormalize_lab(output.data.double().cpu()), color_space)
        plt.figure()
        plt.imshow(np.transpose(out_img[0],(1, 2, 0)))
        plt.axis('off')
        plt.savefig('/Users/spuliz/Desktop/schedio/img/output/output.jpg')
        return send_from_directory(app.config["CLIENT_IMAGES"], filename='output.jpg', as_attachment=True)
        return redirect(url_for('http://127.0.0.1:5000/profit/output.jpg'))
    return render_template('profit_cloth.html')