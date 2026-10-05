# -*- coding: utf-8 -*-
"""
Escáner de certificados de materiales
-------------------------------------
Servidor que corre en una PC de la red de la empresa. Los celulares entran
desde Chrome a  http://IP-DE-LA-PC:5050 , toman la foto de cada hoja y el
servidor la endereza, la limpia, arma el PDF y lo guarda en las carpetas
configuradas en config.json.

Uso (Git Bash):
    pip install -r requirements.txt
    python servidor.py
"""
import base64
import json
import re
import shutil
import socket
import time
import urllib.error
import urllib.request
import uuid
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file

BASE = Path(__file__).resolve().parent
TEMPORAL = BASE / "_temporal"      # hojas en proceso (se borran al crear el PDF)
PENDIENTES = BASE / "_pendientes"  # PDFs que no se pudieron copiar a ningún destino
ARCHIVO_CONFIG = BASE / "config.json"

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]


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
    # Subcarpetas dentro de cada destino. Variables: {anio} {mes} {dia} {proveedor}
    "subcarpetas": "{anio}/{mes}",
    "puerto": 5050,
    # Tamaño del lado largo de cada hoja en píxeles (2480 = A4 a ~210 dpi).
    "lado_maximo_px": 2480,
    "calidad_jpg": 85,
    # "A4": si la hoja mide parecido a un A4, se entrega con la proporción exacta de A4
    # (corrige la deformación de las fotos tomadas inclinadas). "libre": no se ajusta.
    "formato_hoja": "A4",
    # Opcional: lectura automática de los datos del certificado con Gemini.
    # Si se deja vacío la app funciona igual, solo sin el botón de IA.
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


def limpiar_nombre(texto, maximo=80):
    texto = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", str(texto or ""))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:maximo].strip()


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
# Rutas
# --------------------------------------------------------------------------
@app.get("/")
def inicio():
    respuesta = send_file(BASE / "index.html")
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


@app.get("/manifest.json")
def manifiesto():
    return jsonify({
        "name": "Escáner de certificados", "short_name": "Escáner",
        "start_url": "/", "display": "standalone",
        "background_color": "#f4f5f2", "theme_color": "#1f3d2b",
        "icons": [{"src": "/icono.png", "sizes": "192x192", "type": "image/png"}],
    })


@app.get("/icono.png")
def icono():
    lienzo = np.full((192, 192, 3), (43, 61, 31), np.uint8)
    cv2.rectangle(lienzo, (56, 38), (136, 154), (255, 255, 255), -1)
    for y in (66, 86, 106, 126):
        cv2.line(lienzo, (70, y), (122, y), (43, 61, 31), 5)
    _, datos = cv2.imencode(".png", lienzo)
    return Response(datos.tobytes(), mimetype="image/png")


@app.get("/api/config")
def api_config():
    return jsonify({
        "ia": bool(CONFIG.get("gemini_api_key")),
        "destinos": CONFIG["carpetas_destino"],
    })


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


@app.get("/api/pagina/<sesion>/<pagina>.jpg")
def api_pagina(sesion, pagina):
    carpeta = carpeta_sesion(sesion)
    sufijo = "_orig" if request.args.get("original") else ""
    ruta = carpeta / f"{validar_id(pagina)}{sufijo}.jpg"
    if not ruta.exists():
        abort(404)
    if request.args.get("mini"):
        imagen = leer_imagen(ruta)
        factor = 420.0 / max(imagen.shape[:2])
        imagen = cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        _, datos = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(datos.tobytes(), mimetype="image/jpeg")
    return send_file(ruta, mimetype="image/jpeg")


@app.delete("/api/pagina/<sesion>/<pagina>")
def api_borrar(sesion, pagina):
    carpeta = carpeta_sesion(sesion)
    pagina = validar_id(pagina)
    for sufijo in ("_orig.jpg", ".jpg", ".json"):
        (carpeta / f"{pagina}{sufijo}").unlink(missing_ok=True)
    return jsonify({"ok": True})


@app.post("/api/pdf")
def api_pdf():
    """Une las hojas en un PDF y lo copia a cada carpeta destino."""
    cuerpo = request.get_json(force=True)
    carpeta = carpeta_sesion(cuerpo.get("sesion"))
    ids = [validar_id(p) for p in cuerpo.get("paginas", [])]
    if not ids:
        abort(400, "No hay hojas para el PDF")

    paginas = []
    for pagina in ids:
        jpg, info = carpeta / f"{pagina}.jpg", carpeta / f"{pagina}.json"
        if not jpg.exists() or not info.exists():
            abort(404, "Falta una hoja en el servidor; revisa la lista")
        paginas.append({"jpg": jpg.read_bytes(),
                        **json.loads(info.read_text(encoding="utf-8"))})

    ahora = datetime.now()
    proveedor = limpiar_nombre(cuerpo.get("proveedor"), 60)
    nombre = limpiar_nombre(cuerpo.get("nombre"), 90)
    partes = [p for p in (proveedor, nombre) if p]
    if partes:
        archivo = " - ".join(partes + [ahora.strftime("%Y-%m-%d")])
    else:
        archivo = "Certificado " + ahora.strftime("%Y-%m-%d %H%M%S")

    pdf = crear_pdf(paginas, archivo)

    variables = {"anio": ahora.strftime("%Y"),
                 "mes": f"{ahora.month:02d} - {MESES[ahora.month - 1]}",
                 "dia": ahora.strftime("%d"),
                 "proveedor": proveedor or "Sin proveedor"}
    try:
        sub = str(CONFIG.get("subcarpetas", "")).format(**variables)
    except (KeyError, IndexError, ValueError):
        sub = f"{variables['anio']}/{variables['mes']}"
    # Windows no acepta carpetas que terminen en punto o espacio
    sub_partes = [limpiar_nombre(p).rstrip(" .") for p in re.split(r"[\\/]+", sub)]
    sub_partes = [p for p in sub_partes if p]

    guardados, errores = [], []
    for destino in CONFIG["carpetas_destino"]:
        try:
            final = Path(destino).joinpath(*sub_partes)
            final.mkdir(parents=True, exist_ok=True)
            ruta = ruta_libre(final, archivo)
            ruta.write_bytes(pdf)
            guardados.append(str(ruta))
        except Exception as error:
            errores.append({"destino": destino, "error": str(error)})

    if not guardados:
        # Ningún destino respondió: el PDF no se pierde, queda junto al servidor
        PENDIENTES.mkdir(exist_ok=True)
        ruta = ruta_libre(PENDIENTES, archivo)
        ruta.write_bytes(pdf)
        return jsonify({"ok": False, "pendiente": str(ruta), "errores": errores,
                        "archivo": ruta.name}), 507

    shutil.rmtree(carpeta, ignore_errors=True)
    print(f"[PDF] {Path(guardados[0]).name}  ({len(paginas)} hojas, {len(pdf) // 1024} KB)")
    return jsonify({"ok": True, "archivo": Path(guardados[0]).name, "hojas": len(paginas),
                    "kb": len(pdf) // 1024, "guardados": guardados, "errores": errores})


# --------------------------------------------------------------------------
# IA opcional (Gemini): lee los datos del certificado para proponer el nombre
# --------------------------------------------------------------------------
INSTRUCCION_IA = (
    "Esta imagen es un certificado de un material (certificado de calidad, de análisis, "
    "de conformidad o similar). Extrae los datos y responde SOLO un JSON con estas claves: "
    '"proveedor" (empresa que emite el certificado o fabrica el material), '
    '"material" (nombre corto del producto o material), '
    '"numero_certificado", "lote", "fecha" (formato AAAA-MM-DD). '
    "Copia los valores tal como aparecen en el documento. Si un dato no aparece o no se "
    'lee con claridad, deja esa clave como "". No inventes nada.'
)


def consultar_gemini(jpg):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{CONFIG['gemini_modelo']}:generateContent")
    cuerpo = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(jpg).decode("ascii")}},
            {"text": INSTRUCCION_IA},
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
    return {clave: str(resultado.get(clave) or "").strip()
            for clave in ("proveedor", "material", "numero_certificado", "lote", "fecha")}


@app.post("/api/ia")
def api_ia():
    if not CONFIG.get("gemini_api_key"):
        abort(400, "La IA no está configurada")
    cuerpo = request.get_json(force=True)
    carpeta = carpeta_sesion(cuerpo.get("sesion"))
    ruta = carpeta / f"{validar_id(cuerpo.get('id'))}.jpg"
    if not ruta.exists():
        abort(404, "No se encontró la hoja")
    imagen = leer_imagen(ruta)
    factor = min(1.0, 1800.0 / max(imagen.shape[:2]))
    if factor < 1:
        imagen = cv2.resize(imagen, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        return jsonify({"ok": True, **consultar_gemini(jpg.tobytes())})
    except urllib.error.HTTPError as error:
        detalle = error.read().decode("utf-8", "ignore")[:300]
        return jsonify({"ok": False, "error": f"Gemini respondió {error.code}: {detalle}"}), 502
    except Exception as error:
        return jsonify({"ok": False, "error": f"No se pudo consultar a Gemini: {error}"}), 502


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
    print("=" * 62)
    print("  ESCÁNER DE CERTIFICADOS")
    print(f"  En los celulares abre:  http://{ip_local()}:{puerto}")
    print("  Los PDF se guardan en:")
    for destino in CONFIG["carpetas_destino"]:
        print(f"     - {destino}")
    print(f"  Lectura con IA (Gemini): {'activada' if CONFIG.get('gemini_api_key') else 'apagada'}")
    print("  Para detener: Ctrl + C")
    print("=" * 62)
    app.run(host="0.0.0.0", port=puerto, threaded=True)
