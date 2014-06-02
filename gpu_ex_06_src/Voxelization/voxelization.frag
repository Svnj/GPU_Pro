// Extension aktivieren, damit << im Befehlssatz vorliegt.
#extension GL_EXT_gpu_shader4 : enable

// Ausgabevariable
varying out uvec4 result;

void main()
{	
	// TODO: Tiefenwert von [0..1] auf {0..127} abbilden.
	unsigned int depth = unsigned int(gl_FragCoord.z * 127.0f);

	// Dies ergibt beispielsweise den Wert 42.
	// Erzeugen Sie nun eine bit-Maske, in der das (im Beispiel) 42te Bit (von rechts gez�hlt) eine 1 ist und alle anderen eine 0.
	// 00000000..000000010000000..00000000
	// |<- 86 Nullen ->| |<- 41 Nullen ->|
	//                  ^
	//                Bit 42
	// Weisen Sie diese bit-Maske der Variable 'result' zu.

	result.w = 0u;
	result.z = 0u;
	result.y = 0u;
	result.x = 0u;

	if (depth < 32u)
		result.w = 1u << depth;
	else if (depth < 64u)
		result.z = 1u << (depth % 32u);
	else if (depth < 96u)
		result.y = 1u << (depth % 32u);
	else
		result.x = 1u << (depth % 32u);
}
