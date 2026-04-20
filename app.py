import os
import json
import uuid
import random
import math
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

session_images = {}
session_model = {}
session_data = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def img_to_vec(img_bytes, sz, is_rgb):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize((sz, sz), Image.LANCZOS)
    pixels = list(img.getdata())
    vec = []
    if is_rgb:
        for r, g, b in pixels:
            vec.extend([r/255.0, g/255.0, b/255.0])
    else:
        for r, g, b in pixels:
            gray = (0.299*r + 0.587*g + 0.114*b) / 255.0
            vec.append(gray)
    return vec

def col_mean(matrix):
    N = len(matrix)
    D = len(matrix[0])
    mean = [0.0] * D
    for row in matrix:
        for j in range(D):
            mean[j] += row[j]
    return [x / N for x in mean]

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vec_dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def vec_norm(a):
    return math.sqrt(vec_dot(a, a))

def vec_normalise(a):
    n = vec_norm(a)
    if n < 1e-12:
        return a[:]
    return [x/n for x in a]

def gram_matrix(centred):
    N = len(centred)
    G = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            G[i][j] = vec_dot(centred[i], centred[j]) / (N - 1)
    return G

def covariance_matrix(centred):
    N = len(centred)
    D = len(centred[0])
    cov = [[0.0]*D for _ in range(D)]
    for row in centred:
        for i in range(D):
            for j in range(D):
                cov[i][j] += row[i] * row[j]
    for i in range(D):
        for j in range(D):
            cov[i][j] /= (N - 1)
    return cov

def power_iteration(matrix, dim, k, max_iter=250):
    eigenvals = []
    eigenvecs = []
    residual = [row[:] for row in matrix]
    for _ in range(k):
        v = vec_normalise([random.gauss(0, 1) for _ in range(dim)])
        lam = 0.0
        for _ in range(max_iter):
            Av = [0.0] * dim
            for i in range(dim):
                for j in range(dim):
                    Av[i] += residual[i][j] * v[j]
            lam = vec_norm(Av)
            if lam < 1e-12:
                break
            v = [x / lam for x in Av]
        eigenvals.append(lam)
        eigenvecs.append(v)
        for i in range(dim):
            for j in range(dim):
                residual[i][j] -= lam * v[i] * v[j]
    return eigenvals, eigenvecs

def lift_eigenvectors(gram_evecs, centred):
    D = len(centred[0])
    result = []
    for u in gram_evecs:
        dv = [0.0] * D
        for i in range(len(centred)):
            for j in range(D):
                dv[j] += u[i] * centred[i][j]
        result.append(vec_normalise(dv))
    return result

def reconstruct_vec(centred_vec, scores, k, mean, eigenvecs):
    r = mean[:]
    for ci in range(k):
        s = scores[ci]
        for j in range(len(mean)):
            r[j] += s * eigenvecs[ci][j]
    return r

def compute_mse(orig, recon):
    return sum((orig[j] - recon[j])**2 for j in range(len(orig))) / len(orig)

def vec_to_image_b64(vec, sz, is_rgb):
    img = Image.new('RGB', (sz, sz))
    pixels = []
    if is_rgb:
        for p in range(sz * sz):
            r = max(0, min(255, int(vec[p*3]   * 255)))
            g = max(0, min(255, int(vec[p*3+1] * 255)))
            b = max(0, min(255, int(vec[p*3+2] * 255)))
            pixels.append((r, g, b))
    else:
        for p in range(sz * sz):
            g = max(0, min(255, int(vec[p] * 255)))
            pixels.append((g, g, g))
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    sid = request.form.get('sid', 'default')
    if sid not in session_images:
        session_images[sid] = []
    added = []
    for f in files:
        if f and f.filename and allowed_file(f.filename):
            raw = f.read()
            if len(raw) == 0:
                continue
            try:
                img_check = Image.open(io.BytesIO(raw))
                img_check.verify()
            except Exception:
                continue
            fname = secure_filename(f.filename)
            uid = str(uuid.uuid4())[:8]
            img_obj = Image.open(io.BytesIO(raw)).convert('RGB')
            thumb_buf = io.BytesIO()
            thumb = img_obj.copy()
            thumb.thumbnail((80, 80))
            thumb.save(thumb_buf, format='PNG')
            thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()
            session_images[sid].append({
                'id': uid,
                'name': fname,
                'raw': raw,
                'thumb': thumb_b64
            })
            added.append({'id': uid, 'name': fname, 'thumb': thumb_b64})
    return jsonify({'added': added, 'total': len(session_images[sid])})

@app.route('/clear', methods=['POST'])
def clear():
    sid = request.json.get('sid', 'default')
    session_images[sid] = []
    session_model.pop(sid, None)
    session_data.pop(sid, None)
    return jsonify({'ok': True})

@app.route('/remove', methods=['POST'])
def remove():
    sid = request.json.get('sid', 'default')
    img_id = request.json.get('id')
    if sid in session_images:
        session_images[sid] = [x for x in session_images[sid] if x['id'] != img_id]
    return jsonify({'total': len(session_images.get(sid, []))})

@app.route('/split', methods=['POST'])
def split():
    data = request.json
    sid = data.get('sid', 'default')
    pct = data.get('pct', 80) / 100.0
    strategy = data.get('strategy', 'random')
    imgs = session_images.get(sid, [])
    N = len(imgs)
    if N < 2:
        return jsonify({'error': 'Need at least 2 images'}), 400
    indices = list(range(N))
    if strategy == 'random':
        random.shuffle(indices)
    n_train = max(1, round(N * pct))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    if sid not in session_data:
        session_data[sid] = {}
    session_data[sid]['train_idx'] = train_idx
    session_data[sid]['test_idx'] = test_idx
    train_names = [imgs[i]['name'] for i in train_idx]
    test_names = [imgs[i]['name'] for i in test_idx]
    train_thumbs = [imgs[i]['thumb'] for i in train_idx]
    test_thumbs = [imgs[i]['thumb'] for i in test_idx]
    return jsonify({
        'train_idx': train_idx, 'test_idx': test_idx,
        'train_names': train_names, 'test_names': test_names,
        'train_thumbs': train_thumbs, 'test_thumbs': test_thumbs
    })

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    sid = data.get('sid', 'default')
    sz = int(data.get('sz', 32))
    is_rgb = data.get('is_rgb', False)
    k = int(data.get('k', 10))
    center = data.get('center', True)
    imgs = session_images.get(sid, [])
    sd = session_data.get(sid, {})
    train_idx = sd.get('train_idx', [])
    test_idx = sd.get('test_idx', [])
    if len(train_idx) < 2:
        return jsonify({'error': 'Confirm train/test split first'}), 400
    all_vecs = [img_to_vec(im['raw'], sz, is_rgb) for im in imgs]
    D = len(all_vecs[0])
    train_vecs = [all_vecs[i] for i in train_idx]
    N = len(train_vecs)
    mean = col_mean(train_vecs) if center else [0.0] * D
    centred_train = [vec_sub(v, mean) if center else v[:] for v in train_vecs]
    k_safe = min(k, N - 1)
    if D > N:
        G = gram_matrix(centred_train)
        eigenvals, gram_evecs = power_iteration(G, N, k_safe)
        eigenvecs = lift_eigenvectors(gram_evecs, centred_train)
    else:
        cov = covariance_matrix(centred_train)
        eigenvals, eigenvecs = power_iteration(cov, D, k_safe)
    eigenvals = [abs(v) for v in eigenvals]
    total_var = sum(eigenvals) or 1.0
    var_ratios = [v / total_var for v in eigenvals]
    cum_var = []
    running = 0.0
    for v in var_ratios:
        running += v
        cum_var.append(running)
    all_centred = [vec_sub(v, mean) if center else v[:] for v in all_vecs]
    all_scores = [[vec_dot(c, ev) for ev in eigenvecs] for c in all_centred]
    model = {
        'sz': sz, 'is_rgb': is_rgb, 'k': len(eigenvals), 'D': D, 'N': N,
        'mean': mean, 'eigenvecs': eigenvecs, 'eigenvals': eigenvals,
        'var_ratios': var_ratios, 'cum_var': cum_var,
        'all_scores': all_scores, 'all_centred': all_centred,
        'train_idx': train_idx, 'test_idx': test_idx,
        'all_vecs': all_vecs
    }
    session_model[sid] = model
    final_var = cum_var[min(k, len(cum_var)) - 1] * 100
    return jsonify({
        'k': len(eigenvals),
        'D': D, 'N': N,
        'var_ratios': var_ratios,
        'cum_var': cum_var,
        'eigenvals': eigenvals,
        'final_var': round(final_var, 2),
        'top_var': round(var_ratios[0] * 100, 2)
    })

@app.route('/eigenfaces', methods=['POST'])
def eigenfaces():
    sid = request.json.get('sid', 'default')
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    n_show = min(model['k'], 15)
    result = []
    for ci in range(n_show):
        ev = model['eigenvecs'][ci]
        mn = min(ev)
        mx = max(ev)
        rng = mx - mn or 1
        normalised = [(x - mn) / rng for x in ev]
        b64 = vec_to_image_b64(normalised if model['is_rgb'] else normalised,
                               model['sz'], model['is_rgb'])
        result.append({'pc': ci + 1, 'image': b64})
    return jsonify({'eigenfaces': result})

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    data = request.json
    sid = data.get('sid', 'default')
    img_idx = int(data.get('img_idx', 0))
    n_comp = int(data.get('n_comp', 10))
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    orig = model['all_vecs'][img_idx]
    centred = model['all_centred'][img_idx]
    scores = model['all_scores'][img_idx]
    recon = reconstruct_vec(centred, scores, n_comp, model['mean'], model['eigenvecs'])
    residual = [min(1.0, abs(orig[j] - recon[j]) * 5) for j in range(len(orig))]
    mse = compute_mse(orig, recon)
    sz = model['sz']
    is_rgb = model['is_rgb']
    return jsonify({
        'orig_b64':   vec_to_image_b64(orig,     sz, is_rgb),
        'recon_b64':  vec_to_image_b64(recon,    sz, is_rgb),
        'resid_b64':  vec_to_image_b64(residual, sz, is_rgb),
        'mse': round(mse, 8),
        'dims': f'{sz}x{sz} · {model["D"]} dims'
    })

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    sid = data.get('sid', 'default')
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    train_mses = []
    for i in model['train_idx']:
        orig = model['all_vecs'][i]
        recon = reconstruct_vec(model['all_centred'][i], model['all_scores'][i],
                                model['k'], model['mean'], model['eigenvecs'])
        train_mses.append(compute_mse(orig, recon))
    test_mses = []
    for i in model['test_idx']:
        orig = model['all_vecs'][i]
        recon = reconstruct_vec(model['all_centred'][i], model['all_scores'][i],
                                model['k'], model['mean'], model['eigenvecs'])
        test_mses.append(compute_mse(orig, recon))
    avg_train = sum(train_mses) / len(train_mses)
    avg_test = sum(test_mses) / len(test_mses) if test_mses else None
    step = max(1, model['k'] // 20)
    k_values, k_train, k_test = [], [], []
    for ki in range(1, model['k'] + 1, step):
        k_values.append(ki)
        tr = []
        for i in model['train_idx']:
            r = reconstruct_vec(model['all_centred'][i], model['all_scores'][i],
                                ki, model['mean'], model['eigenvecs'])
            tr.append(compute_mse(model['all_vecs'][i], r))
        k_train.append(sum(tr) / len(tr))
        if model['test_idx']:
            ts = []
            for i in model['test_idx']:
                r = reconstruct_vec(model['all_centred'][i], model['all_scores'][i],
                                    ki, model['mean'], model['eigenvecs'])
                ts.append(compute_mse(model['all_vecs'][i], r))
            k_test.append(sum(ts) / len(ts))
    imgs = session_images.get(sid, [])
    bar_labels = (
        [imgs[i]['name'][:12] for i in model['train_idx']] +
        [imgs[i]['name'][:12] for i in model['test_idx']]
    )
    bar_data = train_mses + test_mses
    bar_colors = (['#2563EB'] * len(train_mses)) + (['#16A34A'] * len(test_mses))
    gap = None
    verdict = 'N/A'
    verdict_class = ''
    if avg_test is not None:
        gap = round((avg_test - avg_train) / avg_train * 100, 1)
        if gap < 10:
            verdict, verdict_class = 'Good fit', 'verdict-good'
        elif gap < 40:
            verdict, verdict_class = 'Slight overfit', 'verdict-warn'
        else:
            verdict, verdict_class = 'Overfitting', 'verdict-bad'
    return jsonify({
        'avg_train': avg_train, 'avg_test': avg_test,
        'gap': gap, 'verdict': verdict, 'verdict_class': verdict_class,
        'bar_labels': bar_labels, 'bar_data': bar_data, 'bar_colors': bar_colors,
        'k_values': k_values, 'k_train': k_train, 'k_test': k_test,
        'k': model['k']
    })

@app.route('/anomaly_scores', methods=['POST'])
def anomaly_scores():
    sid = request.json.get('sid', 'default')
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    imgs = session_images.get(sid, [])
    scores = []
    for i in range(len(model['all_vecs'])):
        recon = reconstruct_vec(model['all_centred'][i], model['all_scores'][i],
                                model['k'], model['mean'], model['eigenvecs'])
        scores.append(compute_mse(model['all_vecs'][i], recon))
    train_scores = [scores[i] for i in model['train_idx']]
    mean_mse = sum(train_scores) / len(train_scores)
    labels = [imgs[i]['name'][:14] for i in range(len(imgs))]
    colors = ['#2563EB' if i in model['train_idx'] else '#16A34A'
              for i in range(len(imgs))]
    return jsonify({
        'scores': scores, 'labels': labels, 'colors': colors,
        'mean_mse': mean_mse,
        'train_idx': model['train_idx'],
        'test_idx': model['test_idx']
    })

@app.route('/test_anomaly', methods=['POST'])
def test_anomaly():
    sid = request.form.get('sid', 'default')
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    f = request.files.get('image')
    if not f or not f.filename:
        return jsonify({'error': 'No file'}), 400
    raw = f.read()
    if len(raw) == 0:
        return jsonify({'error': 'Empty file'}), 400
    try:
        vec = img_to_vec(raw, model['sz'], model['is_rgb'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    centred = vec_sub(vec, model['mean'])
    scores = [vec_dot(centred, ev) for ev in model['eigenvecs']]
    recon = reconstruct_vec(centred, scores, model['k'], model['mean'], model['eigenvecs'])
    mse = compute_mse(vec, recon)
    residual = [min(1.0, abs(vec[j] - recon[j]) * 5) for j in range(len(vec))]
    train_scores_mse = []
    for i in model['train_idx']:
        r = reconstruct_vec(model['all_centred'][i], model['all_scores'][i],
                            model['k'], model['mean'], model['eigenvecs'])
        train_scores_mse.append(compute_mse(model['all_vecs'][i], r))
    mean_mse = sum(train_scores_mse) / len(train_scores_mse)
    return jsonify({
        'mse': mse, 'mean_mse': mean_mse,
        'ratio': round(mse / mean_mse, 3),
        'orig_b64': vec_to_image_b64(vec,      model['sz'], model['is_rgb']),
        'recon_b64': vec_to_image_b64(recon,   model['sz'], model['is_rgb']),
        'resid_b64': vec_to_image_b64(residual, model['sz'], model['is_rgb']),
        'filename': secure_filename(f.filename)
    })

@app.route('/projection', methods=['POST'])
def projection():
    data = request.json
    sid = data.get('sid', 'default')
    xi = int(data.get('xi', 0))
    yi = int(data.get('yi', 1))
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    imgs = session_images.get(sid, [])
    pts = []
    for i, scores in enumerate(model['all_scores']):
        pts.append({
            'x': round(scores[xi], 5),
            'y': round(scores[yi], 5),
            'name': imgs[i]['name'],
            'is_train': i in model['train_idx']
        })
    return jsonify({'points': pts, 'k': model['k']})

@app.route('/save_model', methods=['POST'])
def save_model():
    sid = request.json.get('sid', 'default')
    model = session_model.get(sid)
    if not model:
        return jsonify({'error': 'No model'}), 400
    export = {
        'version': 2,
        'sz': model['sz'], 'is_rgb': model['is_rgb'],
        'k': model['k'], 'D': model['D'],
        'mean': model['mean'],
        'eigenvecs': model['eigenvecs'],
        'eigenvals': model['eigenvals'],
        'var_ratios': model['var_ratios'],
        'cum_var': model['cum_var']
    }
    return jsonify(export)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    f = request.files.get('model_file')
    sid = request.form.get('sid', 'default')
    if not f:
        return jsonify({'error': 'No file'}), 400
    try:
        data = json.loads(f.read().decode('utf-8'))
        if 'eigenvecs' not in data or 'mean' not in data:
            return jsonify({'error': 'Invalid model file'}), 400
        model = {
            'sz': data['sz'], 'is_rgb': data['is_rgb'],
            'k': data['k'], 'D': data['D'], 'N': 0,
            'mean': data['mean'], 'eigenvecs': data['eigenvecs'],
            'eigenvals': data['eigenvals'],
            'var_ratios': data['var_ratios'], 'cum_var': data['cum_var'],
            'all_scores': [], 'all_centred': [],
            'train_idx': [], 'test_idx': [], 'all_vecs': []
        }
        session_model[sid] = model
        return jsonify({'ok': True, 'k': data['k'], 'D': data['D'],
                        'var_ratios': data['var_ratios'], 'cum_var': data['cum_var'],
                        'eigenvals': data['eigenvals']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)
