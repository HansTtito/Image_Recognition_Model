from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def descargar_imagen():
    # Configurar el driver de Chrome
    options = webdriver.ChromeOptions()
    
    # Configurar la carpeta de descarga (opcional)
    prefs = {
        "download.default_directory": "C:\\Users\\hkev2\\OneDrive\\Escritorio\\packages\\Modelo_Reconocimiento_Imagenes\\model\\gatos",
        "download.prompt_for_download": False
    }
    options.add_experimental_option("prefs", prefs)
    
    # Inicializar el navegador
    driver = webdriver.Chrome(options=options)
    
    try:
        # Navegar a la página
        driver.get("https://www.pexels.com/es-es/buscar/gatos/")
        
        imagenes_procesadas = set()
        
        while len(imagenes_procesadas) < 5:  # Loop infinito para scroll
                print(f"Iteración del loop. Imágenes procesadas: {len(imagenes_procesadas)}")
                # Encontrar todos los botones de descarga visibles
                botones_descarga = driver.find_elements(By.CSS_SELECTOR, "DownloadButton_downloadButtonText__04wa_")
                
                # Procesar cada botón de descarga nuevo
                for boton in botones_descarga:
                    # Verificar si ya procesamos esta imagen
                    imagen_id = boton.get_attribute("data-id")  # Ajusta según el atributo que identifique únicamente la imagen
                    print(f"Botones de descarga encontrados: {len(botones_descarga)}")

                    if imagen_id in imagenes_procesadas:
                        print(f"Imagen con ID {imagen_id} ya procesada.")
                        continue
                    
                    try:
                        # Scroll hasta el botón
                        print(f"Procesando botón de {len(boton)}")
                        driver.execute_script("arguments[0].scrollIntoView(true);", boton)
                        time.sleep(1)  # Esperar a que se cargue completamente
                        
                        # Click en el botón de descarga
                        WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(boton)
                        ).click()
                        
                        # Esperar a que se abra la ventana modal
                        time.sleep(1)
                        
                        # Click en el botón de descarga dentro de la modal
                        boton_descarga_modal = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='media-download-button']"))
                        )
                        boton_descarga_modal.click()
                        
                        # Esperar a que inicie la descarga
                        time.sleep(2)
                        
                        # Click fuera de la modal para cerrarla
                        By.ActionChains(driver).move_by_offset(0, 0).click().perform()
                        
                        # Marcar imagen como procesada
                        imagenes_procesadas.add(imagen_id)
                        
                    except Exception as e:
                        print(f"Error al procesar imagen: {e}")
                        continue
                
                # Hacer scroll
                last_height = driver.execute_script("return document.body.scrollHeight")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # Esperar a que carguen nuevas imágenes
                
                # Verificar si llegamos al final de la página
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # Intentar un último scroll para asegurarnos
                    time.sleep(2)
                    if new_height == driver.execute_script("return document.body.scrollHeight"):
                        break
                
    except Exception as e:
        print(f"Error general: {e}")
            
    finally:
        driver.quit()

if __name__ == "__main__":
    descargar_imagen()