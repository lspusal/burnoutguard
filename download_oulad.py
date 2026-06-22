#!/usr/bin/env python3
"""
Script para descargar Open University Learning Analytics Dataset (OULAD)

OULAD es un dataset público que contiene datos de 32,593 estudiantes,
22 cursos, y más de 10 millones de interacciones en plataformas de 
aprendizaje digital.

Fuentes:
- https://analyse.kmi.open.ac.uk/open_dataset
- Kuzilek et al. (2015) - Open University Learning Analytics Dataset

Requisitos:
- ~10GB de espacio en disco
- Conexión a internet estable
- ~30-60 minutos de tiempo de descarga

Uso:
    python download_oulad.py --output data/ --extract True
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OULADDownloader:
    """Clase para descargar y procesar OULAD"""
    
    # URLs de descarga (pueden cambiar - verificar https://analyse.kmi.open.ac.uk/open_dataset)
    DATASET_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"
    
    # Ficheros que se descargarán
    FILES_TO_DOWNLOAD = {
        'studentVLE': 'studentVLE.zip',
        'studentInfo': 'studentInfo.zip',
        'studentAssessment': 'studentAssessment.zip',
        'assessments': 'assessments.zip',
        'courses': 'courses.zip',
        'vle': 'vle.zip'
    }
    
    def __init__(self, output_dir: str = 'data/', extract: bool = True):
        """
        Inicializar descargador
        
        Args:
            output_dir: Directorio donde guardar datos
            extract: Si True, descomprime los zips automáticamente
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extract = extract
        
    def download_file(self, filename: str, url: str) -> bool:
        """
        Descargar un fichero individual
        
        Args:
            filename: Nombre del fichero
            url: URL de descarga
            
        Returns:
            True si descarga exitosa, False en caso contrario
        """
        filepath = self.output_dir / filename
        
        if filepath.exists():
            logger.info(f"✓ {filename} ya existe, saltando descarga")
            return True
        
        try:
            logger.info(f"Descargando {filename}...")
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"✓ {filename} descargado exitosamente")
            return True
        except Exception as e:
            logger.error(f"✗ Error descargando {filename}: {e}")
            return False
    
    def extract_zip(self, zipfile_path: Path) -> bool:
        """
        Extraer archivo ZIP
        
        Args:
            zipfile_path: Ruta del fichero ZIP
            
        Returns:
            True si extracción exitosa
        """
        try:
            with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                logger.info(f"Extrayendo {zipfile_path.name}...")
                zip_ref.extractall(self.output_dir)
            logger.info(f"✓ {zipfile_path.name} extraído")
            return True
        except Exception as e:
            logger.error(f"✗ Error extrayendo {zipfile_path.name}: {e}")
            return False
    
    def download_all(self) -> bool:
        """
        Descargar todos los ficheros de OULAD
        
        Returns:
            True si todas las descargas exitosas
        """
        logger.info("Iniciando descarga de OULAD...")
        logger.info(f"Destino: {self.output_dir.absolute()}")
        logger.info(f"Tamaño total: ~10GB")
        logger.info("-" * 60)
        
        success_count = 0
        for file_key, filename in self.FILES_TO_DOWNLOAD.items():
            url = f"{self.DATASET_URL}/{filename}"
            if self.download_file(filename, url):
                success_count += 1
                
                # Extraer si está activado
                if self.extract:
                    zipfile_path = self.output_dir / filename
                    if zipfile_path.exists():
                        self.extract_zip(zipfile_path)
        
        logger.info("-" * 60)
        logger.info(f"✓ Descarga completada: {success_count}/{len(self.FILES_TO_DOWNLOAD)} ficheros")
        
        return success_count == len(self.FILES_TO_DOWNLOAD)
    
    def verify_download(self) -> bool:
        """
        Verificar que todos los ficheros se descargaron correctamente
        
        Returns:
            True si todos los ficheros están presentes
        """
        logger.info("Verificando integridad de descargas...")
        
        required_files = [
            'studentVLE.csv',
            'studentInfo.csv',
            'studentAssessment.csv',
            'assessments.csv',
            'courses.csv',
            'vle.csv'
        ]
        
        missing_files = []
        for required_file in required_files:
            filepath = self.output_dir / required_file
            if not filepath.exists():
                missing_files.append(required_file)
        
        if missing_files:
            logger.warning(f"✗ Ficheros faltantes: {missing_files}")
            return False
        
        logger.info(f"✓ Todos los ficheros están presentes")
        return True


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Descargar Open University Learning Analytics Dataset (OULAD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Descargar y extraer a directorio data/
    python download_oulad.py --output data/ --extract True
    
    # Solo descargar (sin extraer)
    python download_oulad.py --output /tmp/oulad --extract False
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/',
        help='Directorio de salida (default: data/)'
    )
    parser.add_argument(
        '--extract', '-e',
        type=bool,
        default=True,
        help='Extraer ficheros ZIP automáticamente (default: True)'
    )
    parser.add_argument(
        '--verify', '-v',
        type=bool,
        default=True,
        help='Verificar integridad de descarga (default: True)'
    )
    
    args = parser.parse_args()
    
    # Crear descargador
    downloader = OULADDownloader(
        output_dir=args.output,
        extract=args.extract
    )
    
    # Descargar
    success = downloader.download_all()
    
    # Verificar
    if args.verify:
        verified = downloader.verify_download()
        success = success and verified
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
