from djitellopy import Tello
import cv2


def main():
    # Crear objeto Tello
    tello = Tello()
    # Conectar al Tello
    tello.connect()
    # Iniciar transmisión de video
    tello.streamon()
    # Obtener el objeto de video del Tello
    frame_read = tello.get_frame_read()
    # Bucle principal para mostrar el video
    while True:
        # Obtener el cuadro de imagen
        img = frame_read.frame
        # Mostrar imagen usando OpenCV
        cv2.imshow("Tello Video", img)
        # Espera para el evento de tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Cuando todo está hecho, liberar la captura de video y cerrar ventanas
    cv2.destroyAllWindows()
    # Detener transmisión de video y desconectar
    tello.streamoff()
    tello.end()


if __name__ == '__main__':
    main()
