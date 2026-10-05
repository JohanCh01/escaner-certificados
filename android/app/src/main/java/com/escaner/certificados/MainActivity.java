package com.escaner.certificados;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.InputType;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.Toast;

import java.io.File;

/**
 * Abre la página del servidor (servidor.py) dentro de la app.
 * La app solo aporta dos cosas: recordar la dirección de la PC y abrir
 * la cámara directamente cuando la página pide una foto.
 */
public class MainActivity extends Activity {

    private static final int PEDIR_FOTO = 1;
    private static final int PEDIR_GALERIA = 2;
    private static final String PUERTO = "5050";

    private WebView web;
    private SharedPreferences ajustes;
    private ValueCallback<Uri[]> respuesta;
    private Uri fotoUri;
    private boolean dialogoAbierto;

    @Override
    protected void onCreate(Bundle estado) {
        super.onCreate(estado);
        ajustes = getSharedPreferences("ajustes", MODE_PRIVATE);

        web = new WebView(this);
        setContentView(web);

        WebSettings opciones = web.getSettings();
        opciones.setJavaScriptEnabled(true);
        opciones.setDomStorageEnabled(true);   // la página guarda aquí las hojas en curso
        opciones.setAllowFileAccess(false);
        opciones.setTextZoom(100);

        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView vista, WebResourceRequest pedido, WebResourceError error) {
                if (pedido.isForMainFrame()) {
                    sinConexion();
                }
            }
        });

        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView vista, ValueCallback<Uri[]> devolver,
                                             FileChooserParams parametros) {
                if (respuesta != null) {
                    respuesta.onReceiveValue(null);
                }
                respuesta = devolver;
                try {
                    if (parametros.isCaptureEnabled()) {
                        abrirCamara();
                    } else {
                        abrirGaleria();
                    }
                } catch (Exception error) {
                    respuesta.onReceiveValue(null);
                    respuesta = null;
                    Toast.makeText(MainActivity.this, "No se pudo abrir la cámara",
                            Toast.LENGTH_LONG).show();
                }
                return true;
            }
        });

        String direccion = ajustes.getString("direccion", "");
        if (direccion.isEmpty()) {
            pedirDireccion(false);
        } else {
            web.loadUrl(direccion);
        }
    }

    // ---------------------------------------------------------------- cámara

    private void abrirCamara() {
        File carpeta = Archivos.carpeta(this);
        carpeta.mkdirs();
        // Borra fotos antiguas para no llenar el celular
        File[] anteriores = carpeta.listFiles();
        long limite = System.currentTimeMillis() - 10 * 60 * 1000L;
        if (anteriores != null) {
            for (File anterior : anteriores) {
                if (anterior.lastModified() < limite) {
                    anterior.delete();
                }
            }
        }
        String nombre = "foto_" + System.currentTimeMillis() + ".jpg";
        fotoUri = Uri.parse("content://" + getPackageName() + ".archivos/fotos/" + nombre);

        Intent camara = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        camara.putExtra(MediaStore.EXTRA_OUTPUT, fotoUri);
        camara.setClipData(ClipData.newRawUri("foto", fotoUri));
        camara.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivityForResult(camara, PEDIR_FOTO);
    }

    private void abrirGaleria() {
        Intent galeria = new Intent(Intent.ACTION_GET_CONTENT);
        galeria.setType("image/*");
        galeria.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(galeria, "Elegir foto"), PEDIR_GALERIA);
    }

    @Override
    protected void onActivityResult(int codigo, int resultado, Intent datos) {
        super.onActivityResult(codigo, resultado, datos);
        if (codigo != PEDIR_FOTO && codigo != PEDIR_GALERIA) {
            return;
        }
        if (respuesta == null) {
            return;
        }
        Uri[] elegido = null;
        if (resultado == RESULT_OK) {
            if (codigo == PEDIR_FOTO && fotoUri != null) {
                File foto = Archivos.archivo(this, fotoUri);
                if (foto != null && foto.length() > 0) {
                    elegido = new Uri[]{fotoUri};
                }
            } else if (codigo == PEDIR_GALERIA && datos != null && datos.getData() != null) {
                elegido = new Uri[]{datos.getData()};
            }
        }
        respuesta.onReceiveValue(elegido);
        respuesta = null;
    }

    // ------------------------------------------------------ dirección del servidor

    /** Acepta "192.168.1.50", "192.168.1.50:5050" o "http://192.168.1.50:5050". */
    private static String normalizar(String texto) {
        String direccion = texto.trim().replace(" ", "");
        while (direccion.endsWith("/")) {
            direccion = direccion.substring(0, direccion.length() - 1);
        }
        if (direccion.isEmpty()) {
            return "";
        }
        if (!direccion.startsWith("http://") && !direccion.startsWith("https://")) {
            direccion = "http://" + direccion;
        }
        String sinEsquema = direccion.substring(direccion.indexOf("://") + 3);
        if (!sinEsquema.contains(":") && !sinEsquema.contains("/")) {
            direccion = direccion + ":" + PUERTO;
        }
        return direccion;
    }

    private void pedirDireccion(boolean sePuedeCancelar) {
        if (dialogoAbierto) {
            return;
        }
        dialogoAbierto = true;
        final EditText campo = new EditText(this);
        campo.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        campo.setHint("192.168.1.50:" + PUERTO);
        campo.setText(ajustes.getString("direccion", "").replace("http://", ""));
        campo.setSingleLine(true);

        AlertDialog.Builder dialogo = new AlertDialog.Builder(this)
                .setTitle("Dirección del servidor")
                .setMessage("Escribe la dirección que aparece en la ventana del servidor de la PC.")
                .setView(campo)
                .setCancelable(sePuedeCancelar)
                .setPositiveButton("Guardar", (d, cual) -> {
                    dialogoAbierto = false;
                    String direccion = normalizar(campo.getText().toString());
                    if (direccion.isEmpty()) {
                        pedirDireccion(sePuedeCancelar);
                        return;
                    }
                    ajustes.edit().putString("direccion", direccion).apply();
                    web.loadUrl(direccion);
                })
                .setOnCancelListener(d -> dialogoAbierto = false);
        if (sePuedeCancelar) {
            dialogo.setNegativeButton("Cancelar", (d, cual) -> dialogoAbierto = false);
        }
        dialogo.show();
    }

    private void sinConexion() {
        if (dialogoAbierto || isFinishing()) {
            return;
        }
        dialogoAbierto = true;
        new AlertDialog.Builder(this)
                .setTitle("No se pudo conectar")
                .setMessage("Revisa que el celular esté en el Wi-Fi de la empresa y que la PC "
                        + "tenga el servidor abierto.\n\nDirección: "
                        + ajustes.getString("direccion", ""))
                .setCancelable(false)
                .setPositiveButton("Reintentar", (d, cual) -> {
                    dialogoAbierto = false;
                    web.loadUrl(ajustes.getString("direccion", ""));
                })
                .setNegativeButton("Cambiar dirección", (d, cual) -> {
                    dialogoAbierto = false;
                    pedirDireccion(false);
                })
                .show();
    }

    // ------------------------------------------------------------- botón atrás

    @Override
    public void onBackPressed() {
        if (dialogoAbierto) {
            return;
        }
        dialogoAbierto = true;
        String[] opciones = {"Recargar", "Cambiar dirección del servidor", "Salir de la app"};
        new AlertDialog.Builder(this)
                .setItems(opciones, (d, cual) -> {
                    dialogoAbierto = false;
                    if (cual == 0) {
                        web.reload();
                    } else if (cual == 1) {
                        pedirDireccion(true);
                    } else {
                        finish();
                    }
                })
                .setOnCancelListener(d -> dialogoAbierto = false)
                .show();
    }
}
