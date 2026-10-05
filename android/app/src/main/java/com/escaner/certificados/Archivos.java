package com.escaner.certificados;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * Proveedor mínimo de archivos: le da a la app de cámara un lugar donde
 * escribir la foto (carpeta temporal de la app) sin pedir permisos de
 * almacenamiento. Solo acepta nombres del tipo foto_123456.jpg.
 */
public class Archivos extends ContentProvider {

    static File carpeta(Context contexto) {
        return new File(contexto.getCacheDir(), "fotos");
    }

    static File archivo(Context contexto, Uri uri) {
        String nombre = uri.getLastPathSegment();
        if (nombre == null || !nombre.matches("foto_\\d+\\.jpg")) {
            return null;
        }
        return new File(carpeta(contexto), nombre);
    }

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String modo) throws FileNotFoundException {
        File destino = archivo(getContext(), uri);
        if (destino == null) {
            throw new FileNotFoundException("Archivo no permitido");
        }
        destino.getParentFile().mkdirs();
        return ParcelFileDescriptor.open(destino, ParcelFileDescriptor.parseMode(modo));
    }

    @Override
    public Cursor query(Uri uri, String[] columnas, String filtro, String[] valoresFiltro,
                        String orden) {
        File destino = archivo(getContext(), uri);
        String[] pedidas = columnas != null
                ? columnas
                : new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE};
        Object[] fila = new Object[pedidas.length];
        for (int i = 0; i < pedidas.length; i++) {
            if (OpenableColumns.DISPLAY_NAME.equals(pedidas[i])) {
                fila[i] = destino != null ? destino.getName() : null;
            } else if (OpenableColumns.SIZE.equals(pedidas[i])) {
                fila[i] = destino != null ? destino.length() : null;
            }
        }
        MatrixCursor cursor = new MatrixCursor(pedidas, 1);
        cursor.addRow(fila);
        return cursor;
    }

    @Override
    public String getType(Uri uri) {
        return "image/jpeg";
    }

    @Override
    public Uri insert(Uri uri, ContentValues valores) {
        return null;
    }

    @Override
    public int delete(Uri uri, String filtro, String[] valoresFiltro) {
        return 0;
    }

    @Override
    public int update(Uri uri, ContentValues valores, String filtro, String[] valoresFiltro) {
        return 0;
    }
}
