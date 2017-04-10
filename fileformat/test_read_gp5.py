# this code was written inspired by this spec: http://dguitar.sourceforge.net/GP4format.html
# also tuxguitar-gtp org.herac.tuxguitar.io.gtp GP5InputStream
# and kguitar-code/convert/convertgtp.cpp

from gp5_reader import GP5File # imports all functions and variables

gp5file = GP5File("test.gp5")

for m in gp5file.measures:
    print(m)
print()
for t in gp5file.tracklist:
    print(t)


print()
for m in range(gp5file.nMeasures):
    for t in range(gp5file.nTracks):
        for v in range(2):  # every track has two voices
            for beat in gp5file.notes[m][t][v]:
                print(beat)
