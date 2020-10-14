CFLAGS = -I. `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`

DEPS = framemod.cpp framemod.hpp


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

% : %.cpp $(DEPS)
	g++ $(CFLAGS) framemod.cpp -o $@ $< $(LIBS)
