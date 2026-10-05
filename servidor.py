# -*- coding: utf-8 -*-
"""
Escáner de certificados de materiales
-------------------------------------
Servidor que corre en una PC de la red de la empresa. Los celulares entran
desde Chrome a  http://IP-DE-LA-PC:5050 , toman la foto de cada hoja y el
servidor la endereza, la limpia, arma el PDF y lo guarda en las carpetas
configuradas en config.json, ordenado como  año / fecha / proveedor / certificado.pdf

Uso (Git Bash):
    pip install -r requirements.txt
    python servidor.py
"""
import base64
import importlib.util
import json
import re
import shutil
import socket
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import zlib
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

BASE = Path(__file__).resolve().parent
TEMPORAL = BASE / "_temporal"      # hojas en proceso (se borran al crear el PDF)
PENDIENTES = BASE / "_pendientes"  # PDFs que no se pudieron copiar a ningún destino
ARCHIVO_CONFIG = BASE / "config.json"
ARCHIVO_PROVEEDORES = BASE / "proveedores.json"   # lista que se administra desde la app


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
def escritorio():
    casa = Path.home()
    for ruta in (casa / "Desktop", casa / "OneDrive" / "Desktop",
                 casa / "Escritorio", casa / "OneDrive" / "Escritorio"):
        if ruta.is_dir():
            return ruta
    return casa


CONFIG_INICIAL = {
    # Carpetas donde se guarda cada PDF. Puedes poner varias: una local y
    # carpetas compartidas de otras PCs, por ejemplo "\\\\PC-CALIDAD2\\Certificados".
    "carpetas_destino": [str(escritorio() / "Certificados de Materiales")],
    # Dentro de cada destino se ordena siempre como: año / fecha / proveedor / certificado.pdf
    "puerto": 5050,
    # Tamaño del lado largo de cada hoja en píxeles (2480 = A4 a ~210 dpi).
    "lado_maximo_px": 2480,
    "calidad_jpg": 85,
    # "A4": si la hoja mide parecido a un A4, se entrega con la proporción exacta de A4
    # (corrige la deformación de las fotos tomadas inclinadas). "libre": no se ajusta.
    "formato_hoja": "A4",
    # El nombre del certificado se lee del encabezado de la hoja 1. Vacío: lo lee esta PC
    # (nada sale de la empresa). Con una clave de Gemini: lo lee Gemini (envía la hoja a Google).
    "gemini_api_key": "",
    "gemini_modelo": "gemini-3.5-flash",
}


def cargar_config():
    config = dict(CONFIG_INICIAL)
    if ARCHIVO_CONFIG.exists():
        try:
            config.update(json.loads(ARCHIVO_CONFIG.read_text(encoding="utf-8")))
        except Exception as error:
            print(f"[!] config.json tiene un error y se ignoró: {error}")
    else:
        ARCHIVO_CONFIG.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


CONFIG = cargar_config()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


# --------------------------------------------------------------------------
# Procesamiento de imagen
# --------------------------------------------------------------------------
def ordenar_esquinas(puntos):
    """Devuelve las 4 esquinas en orden: sup-izq, sup-der, inf-der, inf-izq."""
    puntos = np.array(puntos, dtype=np.float32).reshape(4, 2)
    centro = puntos.mean(axis=0)
    angulos = np.arctan2(puntos[:, 1] - centro[1], puntos[:, 0] - centro[0])
    puntos = puntos[np.argsort(angulos)]          # sentido horario en pantalla
    inicio = int(np.argmin(puntos.sum(axis=1)))   # la más cercana a (0, 0)
    return np.roll(puntos, -inicio, axis=0)


def _cuadrilatero(contorno):
    """Intenta reducir un contorno a 4 esquinas."""
    perimetro = cv2.arcLength(contorno, True)
    for tolerancia in (0.02, 0.03, 0.04, 0.05, 0.07):
        aprox = cv2.approxPolyDP(contorno, tolerancia * perimetro, True)
        if len(aprox) == 4 and cv2.isContourConvex(aprox):
            return aprox.reshape(4, 2).astype(np.float32)
    return None


def detectar_documento(imagen):
    """Busca la hoja en la foto. Devuelve (esquinas 0..1, encontrado)."""
    alto, ancho = imagen.shape[:2]
    escala = 640.0 / max(alto, ancho)
    chica = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    h, w = chica.shape[:2]
    area_total = float(h * w)

    gris = cv2.GaussianBlur(cv2.cvtColor(chica, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Cerrar la imagen borra el texto para que no confunda a la detección
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    liso = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, nucleo, iterations=2)

    candidatos = []  # (puntaje, esquinas)

    def evaluar(contornos, peso):
        for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(contorno)
            if area < 0.12 * area_total:
                break
            cuad = _cuadrilatero(contorno)
            if cuad is None:
                continue
            area_cuad = cv2.contourArea(cuad)
            if 0.12 * area_total < area_cuad < 0.985 * area_total:
                candidatos.append((area_cuad * peso, cuad))

    # 1) Papel claro sobre fondo más oscuro (Otsu)
    _, binaria = cv2.threshold(liso, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    evaluar(contornos, 1.0)

    # 2) Bordes (sirve cuando el fondo es claro, parecido al papel)
    for bajo, alto_umbral in ((40, 120), (15, 50)):
        bordes = cv2.Canny(liso, bajo, alto_umbral)
        bordes = cv2.dilate(bordes, np.ones((3, 3), np.uint8), iterations=2)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        evaluar(contornos, 0.9)

    if not candidatos:
        margen = 0.04
        return [[margen, margen], [1 - margen, margen],
                [1 - margen, 1 - margen], [margen, 1 - margen]], False

    esquinas = ordenar_esquinas(max(candidatos, key=lambda c: c[0])[1])
    esquinas[:, 0] /= w
    esquinas[:, 1] /= h
    return np.clip(esquinas, 0, 1).round(4).tolist(), True


def enderezar(imagen, esquinas_norm, lado_maximo, formato="A4"):
    """Recorta la hoja y corrige la perspectiva."""
    alto, ancho = imagen.shape[:2]
    puntos = ordenar_esquinas(np.array(esquinas_norm, dtype=np.float32) * [ancho, alto])
    si, sd, id_, ii = puntos
    ancho_hoja = max(np.linalg.norm(sd - si), np.linalg.norm(id_ - ii))
    alto_hoja = max(np.linalg.norm(ii - si), np.linalg.norm(id_ - sd))
    if ancho_hoja < 20 or alto_hoja < 20:
        raise ValueError("Las esquinas marcan un área demasiado pequeña")
    if str(formato).upper() == "A4":
        largo, corto = max(ancho_hoja, alto_hoja), min(ancho_hoja, alto_hoja)
        if abs(largo / corto - 1.4142) / 1.4142 < 0.20:
            largo = max(largo, corto * 1.4142)
            corto = largo / 1.4142
            ancho_hoja, alto_hoja = (corto, largo) if alto_hoja >= ancho_hoja else (largo, corto)
    w, h = int(round(ancho_hoja)), int(round(alto_hoja))
    destino = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(puntos.astype(np.float32), destino)
    hoja = cv2.warpPerspective(imagen, matriz, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    if max(w, h) > lado_maximo:
        factor = lado_maximo / max(w, h)
        hoja = cv2.resize(hoja, (int(round(w * factor)), int(round(h * factor))),
                          interpolation=cv2.INTER_AREA)
    return hoja


def _fondo(canal):
    """Estima la iluminación del papel (sin el texto) para quitar sombras."""
    alto, ancho = canal.shape
    escala = 400.0 / max(alto, ancho)
    chica = cv2.resize(canal, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    fondo = cv2.dilate(chica, nucleo)          # elimina trazos oscuros (texto)
    fondo = cv2.medianBlur(fondo, 21)
    fondo = cv2.GaussianBlur(fondo, (0, 0), 6)
    # No dejar que un bloque oscuro grande (logo, encabezado) se tome como "sombra"
    piso = 0.45 * float(np.percentile(fondo, 95))
    fondo = np.maximum(fondo, piso)
    return cv2.resize(fondo, (ancho, alto), interpolation=cv2.INTER_LINEAR)


def mejorar(hoja, filtro):
    """filtro: 'color' (papel blanco, sellos y firmas a color), 'gris' u 'original'."""
    if filtro == "original":
        return hoja
    datos = hoja.astype(np.float32)
    canales = [np.clip(datos[:, :, i] / _fondo(hoja[:, :, i]) * 255.0, 0, 255)
               for i in range(3)]
    plana = cv2.merge(canales)

    # Nitidez suave
    borrosa = cv2.GaussianBlur(plana, (0, 0), 1.2)
    plana = cv2.addWeighted(plana, 1.6, borrosa, -0.6, 0)

    # Niveles: el papel a blanco puro y la tinta más oscura
    gris = cv2.cvtColor(np.clip(plana, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    negro = float(np.clip(np.percentile(gris, 1), 0, 90)) * 0.8
    blanco = 232.0 if filtro == "color" else 225.0
    plana = np.clip((plana - negro) / (blanco - negro) * 255.0, 0, 255).astype(np.uint8)

    if filtro == "gris":
        return cv2.cvtColor(plana, cv2.COLOR_BGR2GRAY)
    return plana


# --------------------------------------------------------------------------
# PDF (escritor mínimo: incrusta los JPG tal cual, sin recomprimir)
# --------------------------------------------------------------------------
A4_CORTO, A4_LARGO = 595.28, 841.89


def _texto_pdf(texto):
    return "<FEFF" + texto.encode("utf-16-be").hex().upper() + ">"


def crear_pdf(paginas, titulo):
    """paginas: lista de dicts {jpg: bytes, ancho, alto, canales}."""
    objetos = {}

    def agregar(numero, cuerpo):
        objetos[numero] = cuerpo if isinstance(cuerpo, bytes) else cuerpo.encode("latin-1")

    total = len(paginas)
    numeros_pagina = [4 + i * 3 for i in range(total)]
    agregar(1, "<< /Type /Catalog /Pages 2 0 R >>")
    agregar(2, "<< /Type /Pages /Count %d /Kids [%s] >>"
            % (total, " ".join(f"{n} 0 R" for n in numeros_pagina)))
    fecha = datetime.now().strftime("D:%Y%m%d%H%M%S")
    agregar(3, "<< /Title %s /Producer %s /CreationDate (%s) >>"
            % (_texto_pdf(titulo), _texto_pdf("Escáner de certificados"), fecha))

    for i, pagina in enumerate(paginas):
        n_pagina, n_contenido, n_imagen = 4 + i * 3, 5 + i * 3, 6 + i * 3
        ancho, alto = pagina["ancho"], pagina["alto"]
        vertical = alto >= ancho
        proporcion = max(ancho, alto) / min(ancho, alto)
        if abs(proporcion - 1.4142) / 1.4142 < 0.12:
            # Tamaño carta o A4: se entrega como hoja A4, centrada
            hoja_w, hoja_h = (A4_CORTO, A4_LARGO) if vertical else (A4_LARGO, A4_CORTO)
        else:
            factor = A4_LARGO / max(ancho, alto)
            hoja_w, hoja_h = ancho * factor, alto * factor
        factor = min(hoja_w / ancho, hoja_h / alto)
        img_w, img_h = ancho * factor, alto * factor
        x, y = (hoja_w - img_w) / 2, (hoja_h - img_h) / 2

        contenido = zlib.compress(
            f"q {img_w:.2f} 0 0 {img_h:.2f} {x:.2f} {y:.2f} cm /Im0 Do Q".encode())
        agregar(n_pagina,
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] "
                "/Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>"
                % (hoja_w, hoja_h, n_imagen, n_contenido))
        agregar(n_contenido,
                b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(contenido)
                + contenido + b"\nendstream")
        espacio = "/DeviceGray" if pagina["canales"] == 1 else "/DeviceRGB"
        cabecera = ("<< /Type /XObject /Subtype /Image /Width %d /Height %d "
                    "/ColorSpace %s /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n"
                    % (ancho, alto, espacio, len(pagina["jpg"]))).encode("latin-1")
        agregar(n_imagen, cabecera + pagina["jpg"] + b"\nendstream")

    salida = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    posiciones = {}
    for numero in sorted(objetos):
        posiciones[numero] = len(salida)
        salida += b"%d 0 obj\n" % numero + objetos[numero] + b"\nendobj\n"
    inicio_xref = len(salida)
    cantidad = len(objetos) + 1
    salida += b"xref\n0 %d\n" % cantidad + b"0000000000 65535 f \n"
    for numero in range(1, cantidad):
        salida += b"%010d 00000 n \n" % posiciones[numero]
    salida += (b"trailer\n<< /Size %d /Root 1 0 R /Info 3 0 R >>\nstartxref\n%d\n%%%%EOF\n"
               % (cantidad, inicio_xref))
    return bytes(salida)


# --------------------------------------------------------------------------
# Utilidades de archivos
# --------------------------------------------------------------------------
def carpeta_sesion(sesion, crear=False):
    if not re.fullmatch(r"[a-f0-9]{8,32}", sesion or ""):
        abort(400, "Sesión inválida")
    ruta = TEMPORAL / sesion
    if crear:
        ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def validar_id(pagina):
    if not re.fullmatch(r"[a-f0-9]{6,32}", pagina or ""):
        abort(400, "Hoja inválida")
    return pagina


def limpiar_nombre(texto, maximo=90, para_carpeta=True):
    """Quita lo que Windows no acepta en nombres de archivo o carpeta."""
    texto = re.sub(r"\s*/\s*", "-", str(texto or ""))          # 000873 / 2026 -> 000873-2026
    texto = re.sub(r'[\\:*?"<>|\r\n\t]+', " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()[:maximo].strip()
    # Windows no acepta nombres que terminen en punto ("S.A.C." se guarda como "S.A.C")
    return texto.rstrip(" .") if para_carpeta else texto


def validar_fecha(texto):
    """Fecha AAAA-MM-DD; si viene vacía se usa el día de hoy."""
    if not texto:
        return date.today()
    try:
        return datetime.strptime(str(texto), "%Y-%m-%d").date()
    except ValueError:
        abort(400, "Fecha inválida")


def carpeta_dia(destino, fecha):
    """Estructura fija: destino / año / fecha / proveedor / certificado.pdf"""
    return Path(destino) / f"{fecha:%Y}" / f"{fecha:%Y-%m-%d}"


def nombre_seguro(texto):
    """Para leer archivos ya guardados: solo un nombre simple, nunca una ruta."""
    texto = str(texto or "")
    if not texto or texto != Path(texto).name or texto.startswith(".") or "\\" in texto:
        abort(400, "Nombre inválido")
    return texto


def destino_de_lectura():
    """Primera carpeta destino disponible (de ahí se lee lo ya escaneado)."""
    for destino in CONFIG["carpetas_destino"]:
        try:
            if Path(destino).is_dir():
                return Path(destino)
        except OSError:
            continue
    return None


def ruta_libre(carpeta, nombre):
    """Evita sobrescribir: agrega (2), (3)... si el archivo ya existe."""
    ruta = carpeta / f"{nombre}.pdf"
    contador = 2
    while ruta.exists():
        ruta = carpeta / f"{nombre} ({contador}).pdf"
        contador += 1
    return ruta


def limpiar_temporales(dias=3):
    if not TEMPORAL.exists():
        return
    limite = time.time() - dias * 86400
    for carpeta in TEMPORAL.iterdir():
        try:
            if carpeta.is_dir() and carpeta.stat().st_mtime < limite:
                shutil.rmtree(carpeta, ignore_errors=True)
        except OSError:
            pass


def leer_imagen(ruta):
    datos = np.fromfile(str(ruta), dtype=np.uint8)   # soporta rutas con tildes en Windows
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# Proveedores (lista compartida por todos los celulares)
# --------------------------------------------------------------------------
_candado_proveedores = threading.Lock()


def leer_proveedores():
    try:
        lista = json.loads(ARCHIVO_PROVEEDORES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    limpios = {limpiar_nombre(p, 60, para_carpeta=False) for p in lista if isinstance(p, str)}
    return sorted((p for p in limpios if p), key=str.casefold)


def guardar_proveedores(lista):
    ARCHIVO_PROVEEDORES.write_text(
        json.dumps(sorted(lista, key=str.casefold), indent=2, ensure_ascii=False),
        encoding="utf-8")


# --------------------------------------------------------------------------
# Páginas básicas
# --------------------------------------------------------------------------
@app.get("/")
def inicio():
    respuesta = send_file(BASE / "index.html")
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


@app.get("/recursos/<nombre>")
def recursos(nombre):
    return send_from_directory(BASE / "recursos", nombre, max_age=3600)


@app.get("/manifest.json")
def manifiesto():
    return jsonify({
        "name": "Escáner de certificados", "short_name": "Escáner",
        "start_url": "/", "display": "standalone",
        "background_color": "#0e3b2a", "theme_color": "#0e3b2a",
        "icons": [{"src": "/recursos/icono-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "/recursos/icono-512.png", "sizes": "512x512", "type": "image/png"}],
    })


@app.get("/api/config")
def api_config():
    return jsonify({
        "ia": motor_ia(),
        "destinos": CONFIG["carpetas_destino"],
        "hoy": date.today().isoformat(),
    })


@app.get("/api/proveedores")
def api_proveedores():
    return jsonify({"proveedores": leer_proveedores()})


@app.post("/api/proveedores")
def api_proveedor_nuevo():
    nombre = limpiar_nombre(request.get_json(force=True).get("nombre"), 60, para_carpeta=False)
    if not limpiar_nombre(nombre):
        abort(400, "Escribe el nombre del proveedor")
    with _candado_proveedores:
        lista = leer_proveedores()
        existente = next((p for p in lista if p.casefold() == nombre.casefold()), None)
        if existente is None:
            lista.append(nombre)
            guardar_proveedores(lista)
        return jsonify({"ok": True, "nombre": existente or nombre,
                        "proveedores": leer_proveedores()})


@app.delete("/api/proveedores")
def api_proveedor_quitar():
    nombre = str(request.get_json(force=True).get("nombre") or "")
    with _candado_proveedores:
        lista = [p for p in leer_proveedores() if p.casefold() != nombre.casefold()]
        guardar_proveedores(lista)
        return jsonify({"ok": True, "proveedores": lista})


# --------------------------------------------------------------------------
# Hojas en proceso
# --------------------------------------------------------------------------
@app.get("/api/sesion/<sesion>")
def api_sesion(sesion):
    carpeta = carpeta_sesion(sesion)
    existentes = [p.stem for p in carpeta.glob("*.json")] if carpeta.exists() else []
    return jsonify({"paginas": existentes})


@app.post("/api/subir")
def api_subir():
    """Recibe la foto, la guarda y devuelve las esquinas detectadas."""
    carpeta = carpeta_sesion(request.form.get("sesion"), crear=True)
    archivo = request.files.get("foto")
    if archivo is None:
        abort(400, "Falta la foto")
    datos = np.frombuffer(archivo.read(), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        abort(400, "No se pudo leer la imagen")
    pagina = uuid.uuid4().hex[:12]
    cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
        str(carpeta / f"{pagina}_orig.jpg"))
    esquinas, encontrado = detectar_documento(imagen)
    return jsonify({"id": pagina, "esquinas": esquinas, "detectado": encontrado})


@app.post("/api/procesar")
def api_procesar():
    """Endereza y limpia la hoja con las esquinas confirmadas por el usuario."""
    cuerpo = request.get_json(force=True)
    carpeta = carpeta_sesion(cuerpo.get("sesion"))
    pagina = validar_id(cuerpo.get("id"))
    original = carpeta / f"{pagina}_orig.jpg"
    if not original.exists():
        abort(404, "La foto ya no está en el servidor; vuelve a tomarla")
    esquinas = cuerpo.get("esquinas")
    filtro = cuerpo.get("filtro", "color")
    if filtro not in ("color", "gris", "original"):
        filtro = "color"
    try:
        hoja = enderezar(leer_imagen(original), esquinas, int(CONFIG["lado_maximo_px"]),
                         CONFIG.get("formato_hoja", "A4"))
    except Exception as error:
        abort(400, f"No se pudo procesar: {error}")
    hoja = mejorar(hoja, filtro)
    giro = int(cuerpo.get("rotar", 0)) % 360
    if giro == 90:
        hoja = cv2.rotate(hoja, cv2.ROTATE_90_CLOCKWISE)
    elif giro == 180:
        hoja = cv2.rotate(hoja, cv2.ROTATE_180)
    elif giro == 270:
        hoja = cv2.rotate(hoja, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imencode(".jpg", hoja, [cv2.IMWRITE_JPEG_QUALITY, int(CONFIG["calidad_jpg"])])[1] \
        .tofile(str(carpeta / f"{pagina}.jpg"))
    info = {"ancho": int(hoja.shape[1]), "alto": int(hoja.shape[0]),
            "canales": 1 if hoja.ndim == 2 else 3}
    (carpeta / f"{pagina}.json").write_text(json.dumps(info), encoding="utf-8")
    return jsonify({"ok": True, **info})


def _miniatura(jpg_o_imagen, lado=420):
    imagen = jpg_o_imagen
    if isinstance(imagen, (bytes, bytearray)):
        imagen = cv2.imdecode(np.frombuffer(imagen, dtype=np.uint8), cv2.IMREAD_COLOR)
    factor = float(lado) / max(imagen.shape[:2])
    imagen = cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    _, datos = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(datos.tobytes(), mimetype="image/jpeg")


@app.get("/api/pagina/<sesion>/<pagina>.jpg")
def api_pagina(sesion, pagina):
    carpeta = carpeta_sesion(sesion)
    sufijo = "_orig" if request.args.get("original") else ""
    ruta = carpeta / f"{validar_id(pagina)}{sufijo}.jpg"
    if not ruta.exists():
        abort(404)
    if request.args.get("mini"):
        return _miniatura(leer_imagen(ruta))
    return send_file(ruta, mimetype="image/jpeg")


@app.delete("/api/pagina/<sesion>/<pagina>")
def api_borrar(sesion, pagina):
    carpeta = carpeta_sesion(sesion)
    pagina = validar_id(pagina)
    for sufijo in ("_orig.jpg", ".jpg", ".json"):
        (carpeta / f"{pagina}{sufijo}").unlink(missing_ok=True)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Guardar el PDF:  destino / año / fecha / proveedor / certificado.pdf
# --------------------------------------------------------------------------
@app.post("/api/pdf")
def api_pdf():
    cuerpo = request.get_json(force=True)
    carpeta = carpeta_sesion(cuerpo.get("sesion"))
    ids = [validar_id(p) for p in cuerpo.get("paginas", [])]
    if not ids:
        abort(400, "No hay hojas para el PDF")
    proveedor = limpiar_nombre(cuerpo.get("proveedor"), 60)
    if not proveedor:
        abort(400, "Elige un proveedor antes de guardar")
    fecha = validar_fecha(cuerpo.get("fecha"))

    paginas = []
    for pagina in ids:
        jpg, info = carpeta / f"{pagina}.jpg", carpeta / f"{pagina}.json"
        if not jpg.exists() or not info.exists():
            abort(404, "Falta una hoja en el servidor; revisa la lista")
        paginas.append({"jpg": jpg.read_bytes(),
                        **json.loads(info.read_text(encoding="utf-8"))})

    archivo = limpiar_nombre(cuerpo.get("nombre"), 90) \
        or "Certificado " + datetime.now().strftime("%H%M%S")
    pdf = crear_pdf(paginas, archivo)

    guardados, errores = [], []
    for destino in CONFIG["carpetas_destino"]:
        try:
            final = carpeta_dia(destino, fecha) / proveedor
            final.mkdir(parents=True, exist_ok=True)
            ruta = ruta_libre(final, archivo)
            ruta.write_bytes(pdf)
            guardados.append(str(ruta))
        except Exception as error:
            errores.append({"destino": destino, "error": str(error)})

    if not guardados:
        # Ningún destino respondió: el PDF no se pierde, queda junto al servidor
        PENDIENTES.mkdir(exist_ok=True)
        ruta = ruta_libre(PENDIENTES, f"{fecha:%Y-%m-%d} - {proveedor} - {archivo}")
        ruta.write_bytes(pdf)
        return jsonify({"ok": False, "pendiente": str(ruta), "errores": errores,
                        "archivo": ruta.name}), 507

    shutil.rmtree(carpeta, ignore_errors=True)
    final = Path(guardados[0])
    print(f"[PDF] {fecha:%Y-%m-%d} / {proveedor} / {final.name}  "
          f"({len(paginas)} hojas, {len(pdf) // 1024} KB)")
    return jsonify({"ok": True, "archivo": final.name, "hojas": len(paginas),
                    "kb": len(pdf) // 1024, "fecha": fecha.isoformat(),
                    "proveedor": proveedor, "errores": errores})


# --------------------------------------------------------------------------
# Consultar lo ya escaneado (calendario)
# --------------------------------------------------------------------------
def _hojas_de_pdf(ruta):
    try:
        with open(ruta, "rb") as archivo:
            cabecera = archivo.read(600)
    except OSError:
        return 0
    encontrado = re.search(rb"/Count (\d+)", cabecera)
    return int(encontrado.group(1)) if encontrado else 0


@app.get("/api/dia")
def api_dia():
    """Certificados guardados en una fecha, agrupados por proveedor."""
    fecha = validar_fecha(request.args.get("fecha"))
    destino = destino_de_lectura()
    proveedores = []
    carpeta = carpeta_dia(destino, fecha) if destino else None
    if carpeta is not None and carpeta.is_dir():
        for sub in sorted((p for p in carpeta.iterdir() if p.is_dir()),
                          key=lambda p: p.name.casefold()):
            archivos = [{"nombre": pdf.name, "hojas": _hojas_de_pdf(pdf),
                         "kb": pdf.stat().st_size // 1024}
                        for pdf in sorted(sub.glob("*.pdf"), key=lambda p: p.stat().st_mtime)]
            if archivos:
                proveedores.append({"nombre": sub.name, "archivos": archivos})
    return jsonify({"fecha": fecha.isoformat(), "proveedores": proveedores,
                    "total": sum(len(p["archivos"]) for p in proveedores)})


@app.get("/api/mes")
def api_mes():
    """Días del mes que tienen certificados (para marcarlos en el calendario)."""
    try:
        anio, mes = int(request.args.get("anio")), int(request.args.get("mes"))
        date(anio, mes, 1)
    except (TypeError, ValueError):
        abort(400, "Mes inválido")
    destino = destino_de_lectura()
    dias = []
    carpeta = destino / f"{anio:04d}" if destino else None
    if carpeta is not None and carpeta.is_dir():
        prefijo = f"{anio:04d}-{mes:02d}-"
        for sub in carpeta.iterdir():
            if sub.is_dir() and sub.name.startswith(prefijo) \
                    and next(sub.glob("*/*.pdf"), None) is not None:
                dias.append(sub.name)
    return jsonify({"dias": sorted(dias)})


@app.get("/api/guardado")
def api_guardado():
    """Devuelve una hoja (imagen) de un PDF ya guardado, para verlo en el celular."""
    fecha = validar_fecha(request.args.get("fecha"))
    proveedor = nombre_seguro(request.args.get("proveedor"))
    archivo = nombre_seguro(request.args.get("archivo"))
    if not archivo.lower().endswith(".pdf"):
        abort(400, "Nombre inválido")
    try:
        numero = int(request.args.get("n", 0))
    except ValueError:
        abort(400, "Hoja inválida")
    destino = destino_de_lectura()
    ruta = carpeta_dia(destino, fecha) / proveedor / archivo if destino else None
    if ruta is None or not ruta.is_file():
        abort(404, "No se encontró el archivo")
    pdf = ruta.read_bytes()
    imagenes = list(re.finditer(rb"/Filter /DCTDecode /Length (\d+) >>\nstream\n", pdf))
    if not 0 <= numero < len(imagenes):
        abort(404, "Este PDF no se puede mostrar aquí; ábrelo desde la PC")
    inicio = imagenes[numero].end()
    jpg = pdf[inicio:inicio + int(imagenes[numero].group(1))]
    if request.args.get("mini"):
        return _miniatura(jpg)
    return Response(jpg, mimetype="image/jpeg")


# --------------------------------------------------------------------------
# IA: lee el encabezado de la hoja 1 para proponer el nombre del certificado
#   - "local":  OCR que corre en esta PC (rapidocr-onnxruntime). Nada sale de la empresa.
#   - "gemini": si config.json tiene gemini_api_key. Lee mejor (tildes, criterio), pero
#               envía la imagen de la hoja a Google.
# --------------------------------------------------------------------------
PALABRAS_TITULO = (
    "CERTIFICADO", "CERTIFICATE", "CERTIFICACION", "CERTIFICATION", "ANALISIS", "ANALYSIS",
    "CALIDAD", "QUALITY", "CONFORMIDAD", "CONFORMITY", "COA", "FICHATECNICA", "DECLARACION",
    "DECLARATION", "INFORME", "REPORT", "CONSTANCIA", "GARANTIA", "HOJADESEGURIDAD",
)
PATRON_NUMERO = re.compile(
    r"(?:\bN\s?[°ºo]\.?|\bN\.|\bNro\.?|\bNumber|\bN[uú]m(?:ero)?\.?|\bC[oó]digo|\bCode|\bFolio|#)"
    r"\s*[:.]?\s*"
    r"([A-Z0-9][A-Z0-9.\-]*(?:\s?/\s?\d+)?)", re.I)

_ocr = None
_candado_ocr = threading.Lock()


def motor_ia():
    if CONFIG.get("gemini_api_key"):
        return "gemini"
    if importlib.util.find_spec("rapidocr_onnxruntime") is not None:
        return "local"
    return ""


def _motor_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def _normalizar(texto):
    sin_tildes = unicodedata.normalize("NFD", str(texto or "").upper())
    return "".join(c for c in sin_tildes if c.isalnum() and ord(c) < 128)


def _separar_palabras(recorte):
    """Devuelve los tramos (x inicial, x final) de cada palabra de una línea de texto."""
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    alto = gris.shape[0]
    _, tinta = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if tinta.mean() > 127:                       # texto claro sobre fondo oscuro
        tinta = 255 - tinta
    ocupada = (tinta > 0).sum(axis=0) > max(1, int(0.04 * alto))
    columnas = np.flatnonzero(ocupada)
    if len(columnas) == 0:
        return [], 0
    x0, x1 = int(columnas[0]), int(columnas[-1]) + 1
    huecos, inicio = [], None
    for x in range(x0, x1):
        if not ocupada[x]:
            if inicio is None:
                inicio = x
        elif inicio is not None:
            huecos.append((inicio, x))
            inicio = None
    if not huecos:
        return [(x0, x1)], 0
    # Un hueco es "espacio entre palabras" si es bastante más ancho que el hueco entre letras
    umbral = max(0.17 * alto, 1.9 * float(np.median([b - a for a, b in huecos])))
    tramos, inicio = [], x0
    for a, b in huecos:
        if b - a >= umbral:
            tramos.append((inicio, a))
            inicio = b
    tramos.append((inicio, x1))
    return tramos, umbral


def _con_espacios(motor, cabecera, linea):
    """El OCR suele pegar las palabras; aquí se recuperan los espacios de una línea."""
    y0, y1 = max(0, linea["y0"] - 4), linea["y1"] + 4
    x0, x1 = max(0, linea["x0"] - 4), linea["x1"] + 4
    recorte = cabecera[y0:y1, x0:x1]
    if recorte.size == 0:
        return linea["texto"].strip()
    tramos, umbral = _separar_palabras(recorte)
    if len(tramos) <= 1:
        return linea["texto"].strip()
    margen = int(max(2, min(6, umbral / 2)))
    palabras = []
    for a, b in tramos:
        trozo = recorte[:, max(0, a - margen):b + margen]
        salida, _ = motor(trozo, use_det=False, use_cls=False)
        # un tramo puede traer más de una palabra si el OCR vio un espacio dentro
        palabras += str(salida[0][0]).split() if salida else []
    original = " ".join(linea["texto"].split())
    if len(palabras) <= len(original.split()):
        return original                       # la lectura de la línea ya traía los espacios
    junto = original.replace(" ", "")
    if sum(len(p) for p in palabras) == len(junto):
        # Mismas letras: se usa la lectura de la línea completa (más fiable) con estos cortes
        partes, i = [], 0
        for palabra in palabras:
            partes.append(junto[i:i + len(palabra)])
            i += len(palabra)
        return " ".join(partes)
    return " ".join(palabras)


def leer_encabezado_local(imagen, proveedor=""):
    motor = _motor_ocr()
    cabecera = imagen[: int(imagen.shape[0] * 0.45)]
    if cabecera.ndim == 2:
        cabecera = cv2.cvtColor(cabecera, cv2.COLOR_GRAY2BGR)
    if cabecera.shape[1] > 1600:
        factor = 1600.0 / cabecera.shape[1]
        cabecera = cv2.resize(cabecera, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    with _candado_ocr:
        resultado, _ = motor(cabecera, use_cls=False)
        lineas = []
        for caja, texto, confianza in resultado or []:
            puntos = np.array(caja, dtype=np.float32)
            lineas.append({
                "texto": str(texto), "conf": float(confianza),
                "x0": int(puntos[:, 0].min()), "x1": int(puntos[:, 0].max()),
                "y0": int(puntos[:, 1].min()), "y1": int(puntos[:, 1].max()),
            })
        lineas.sort(key=lambda l: (l["y0"], l["x0"]))
        prov = _normalizar(proveedor)

        def es_candidata(linea):
            norma = _normalizar(linea["texto"])
            letras = sum(c.isalpha() for c in norma)
            if linea["conf"] < 0.6 or letras < 5 or letras < 0.6 * len(norma):
                return False
            return not (len(prov) >= 4 and (prov in norma or norma in prov))

        candidatas = [l for l in lineas if es_candidata(l)]
        con_clave = [l for l in candidatas
                     if any(p in _normalizar(l["texto"]) for p in PALABRAS_TITULO)]
        grupo = con_clave or candidatas
        if not grupo:
            return {"titulo": "", "numero": ""}
        titular = max(grupo, key=lambda l: l["y1"] - l["y0"])     # la letra más grande
        titulo = _con_espacios(motor, cabecera, titular)

        # El número suele estar en el título o en las líneas que lo rodean
        posicion = lineas.index(titular)
        cercanas = [titular] + lineas[posicion + 1:posicion + 6] + lineas[max(0, posicion - 3):posicion]
        numero = ""
        for linea in cercanas:
            texto = titulo if linea is titular else _con_espacios(motor, cabecera, linea)
            for hallado in PATRON_NUMERO.finditer(texto):
                valor = hallado.group(1).strip(" .-")
                if sum(c.isdigit() for c in valor) >= 2:
                    numero = valor
                    break
            if numero:
                # Si el número estaba dentro del título ("Certificado N° 123"), no se repite
                if linea is titular:
                    titulo = PATRON_NUMERO.sub("", titulo).strip(" .:-")
                break
    return {"titulo": titulo, "numero": numero}


INSTRUCCION_GEMINI = (
    "Esta imagen es la primera hoja de un certificado de un material (certificado de calidad, "
    "de análisis, de conformidad o similar). Lee SOLO el encabezado y responde SOLO un JSON con "
    'dos claves: "titulo" (el título del documento tal como está impreso, por ejemplo '
    '"CERTIFICADO DE ANÁLISIS"; no es el nombre de la empresa) y "numero" (el número o código '
    'del certificado que aparece junto al título; "" si no aparece). Copia los valores tal '
    "como están escritos. No inventes nada."
)


def leer_encabezado_gemini(jpg):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{CONFIG['gemini_modelo']}:generateContent")
    cuerpo = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(jpg).decode("ascii")}},
            {"text": INSTRUCCION_GEMINI},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    peticion = urllib.request.Request(
        url, data=json.dumps(cuerpo).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": CONFIG["gemini_api_key"]})
    with urllib.request.urlopen(peticion, timeout=60) as respuesta:
        datos = json.loads(respuesta.read().decode("utf-8"))
    partes = datos["candidates"][0]["content"]["parts"]
    texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
    coincidencia = re.search(r"\{.*\}", texto, re.S)
    resultado = json.loads(coincidencia.group(0) if coincidencia else texto)
    return {clave: str(resultado.get(clave) or "").strip() for clave in ("titulo", "numero")}


def armar_nombre(titulo, numero):
    titulo, numero = limpiar_nombre(titulo, 70), limpiar_nombre(numero, 30)
    if titulo and numero:
        return f"{titulo} N° {numero}"
    return titulo or numero


@app.post("/api/ia")
def api_ia():
    motor = motor_ia()
    if not motor:
        abort(400, "La lectura automática no está instalada en el servidor")
    cuerpo = request.get_json(force=True)
    carpeta = carpeta_sesion(cuerpo.get("sesion"))
    ruta = carpeta / f"{validar_id(cuerpo.get('id'))}.jpg"
    if not ruta.exists():
        abort(404, "No se encontró la hoja")
    imagen = leer_imagen(ruta)
    aviso = ""
    if motor == "gemini":
        factor = min(1.0, 1800.0 / max(imagen.shape[:2]))
        chica = imagen if factor == 1.0 else cv2.resize(
            imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        _, jpg = cv2.imencode(".jpg", chica, [cv2.IMWRITE_JPEG_QUALITY, 85])
        try:
            leido = leer_encabezado_gemini(jpg.tobytes())
            return jsonify({"ok": True, "origen": "gemini",
                            "nombre": armar_nombre(leido["titulo"], leido["numero"])})
        except Exception as error:
            aviso = f"Gemini no respondió ({error})"
            print(f"[IA] {aviso}")
            if importlib.util.find_spec("rapidocr_onnxruntime") is None:
                return jsonify({"ok": False, "error": aviso}), 502
    try:
        leido = leer_encabezado_local(imagen, cuerpo.get("proveedor") or "")
    except Exception as error:
        return jsonify({"ok": False, "error": f"No se pudo leer el encabezado: {error}"}), 500
    return jsonify({"ok": True, "origen": "local", "aviso": aviso,
                    "nombre": armar_nombre(leido["titulo"], leido["numero"])})


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def error_json(error):
    return jsonify({"ok": False, "error": getattr(error, "description", str(error))}), error.code


def ip_local():
    try:
        conexion = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conexion.connect(("10.255.255.255", 1))
        ip = conexion.getsockname()[0]
        conexion.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    limpiar_temporales()
    puerto = int(CONFIG["puerto"])
    lectura = {"gemini": "Gemini (envía la hoja 1 a Google)",
               "local": "en esta PC (nada sale de la empresa)",
               "": "no instalada (pip install rapidocr-onnxruntime)"}[motor_ia()]
    print("=" * 66)
    print("  ESCÁNER DE CERTIFICADOS")
    print(f"  En los celulares abre:  http://{ip_local()}:{puerto}")
    print("  Los PDF se guardan en:")
    for destino in CONFIG["carpetas_destino"]:
        print(f"     - {destino}")
    print("       ordenados como  año / fecha / proveedor / certificado.pdf")
    print(f"  Lectura del encabezado: {lectura}")
    print(f"  Proveedores cargados: {len(leer_proveedores())}")
    print("  Para detener: Ctrl + C")
    print("=" * 66)
    app.run(host="0.0.0.0", port=puerto, threaded=True)
