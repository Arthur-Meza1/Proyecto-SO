#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

int contador = 0; 

void *incrementar(void *arg) {
    long id = (long) arg;
    for (int i = 0; i < 5; i++) {
        contador++; // acceso no sincronizado
        printf("Hilo %ld incrementa contador = %d\n", id, contador);
        usleep(100000); 
    }
    return NULL;
}

int main() {
    pthread_t h1, h2;

    pthread_create(&h1, NULL, incrementar, (void*)1);
    pthread_create(&h2, NULL, incrementar, (void*)2);

    pthread_join(h1, NULL);
    pthread_join(h2, NULL);

    printf("Valor final del contador = %d\n", contador);
    return 0;
}
