# Generador de Dataset para Carrusel XSelling (MeLi)

Este script construye un dataset de entrenamiento a partir de eventos de interacción con un carrusel de productos/servicios (prints, taps y pagos).  
El objetivo es generar una tabla que incluya las impresiones de la **última semana** junto con métricas históricas de las **3 semanas previas**.

## Entradas requeridas

Los archivos deben estar en el mismo directorio indicado por `--input_dir`:

| Archivo     | Formato  | Descripción                                                                      
|-------------|----------|-----------------------------------------------------------------------------     
| prints.json | JSON Lines | Registros de impresiones del carrusel (día, usuario, value_prop, posición).    
| taps.json   | JSON Lines | Registros de taps/clicks en el carrusel (misma estructura que `prints.json`).  
| pays.csv    | CSV        | Pagos asociados a una `value_prop` (fecha, monto, usuario, value_prop).        

## Salida

| Archivo          | Formato | Descripción 
|------------------|---------|-------------
| dataset_ready.csv| CSV     | Dataset final con impresiones recientes y métricas históricas. 

## Instalación

Requiere **Python 3.8+** y la librería `pandas`.

```bash
pip install pandas
