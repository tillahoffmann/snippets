#include <signal.h>
#include <string.h>

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "SIGTERM") == 0) {
            signal(SIGTERM, SIG_IGN);
        } else {
            return 42;
        }
    }

    while (1) {
        // do nothing busily ...
    }
}
