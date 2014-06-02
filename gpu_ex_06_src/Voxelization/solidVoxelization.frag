// Extension aktivieren, damit << im Befehlssatz vorliegt.
#extension GL_EXT_gpu_shader4 : enable

// Ausgabevariable
varying out uvec4 result;

void main()
{	
	unsigned int depth = unsigned int(gl_FragCoord.z * 127.0f);

	result.w = 0u;
	result.z = 0u;
	result.y = 0u;
	result.x = 0u;

	// Todo: Wieso funktioniert das? [erster Eindruck: Fülle ALLES hinter sichtbarem Voxel -> Fake?]
	// Todo: Und wieso nicht fuer Teapot?
	// Wollen wir das nehmen, oder etwas anderes implementieren?
	if (depth < 32u)
	{
		result.w = ~0u << depth;
		result.z = ~0u;
		result.y = ~0u;
		result.x = ~0u;
	}
	else if (depth < 64u)
	{
		result.z = ~0u << (depth % 32u);
		result.y = ~0u;
		result.x = ~0u;
	}
	else if (depth < 96u)
	{
		result.y = ~0u << (depth % 32u);
		result.x = ~0u;
	}
	else
		result.x = ~0u << (depth % 32u);
}
