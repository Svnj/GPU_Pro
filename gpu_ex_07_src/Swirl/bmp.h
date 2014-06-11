#ifndef __BMP_LOADER__
#define __BMP_LOADER__

#include <fstream>
#include <iostream>
#include <cstdlib>

class Bitmap
{
	private:
		unsigned int width;		// Width of image
		unsigned int height;	// Height of image
		unsigned char* data;	// Data stored in image (rgb)
		unsigned short planes;	// number of planes in image (must be 1) 	
		unsigned short bpp;		// bits per pixel (24)

	public:

		Bitmap(const char *fname) : width(0), height(0), data(0), planes(0), bpp(0)
		{
			using namespace std;		
			ifstream fin(fname, ios::in | ios::binary);	
			if( !fin )
			{
				cerr << "File not found " << fname << '\n';
				return;
			}  	
			fin.seekg(18, ios::cur);
			fin.read((char *)&width, sizeof(unsigned));
			fin.read((char *)&height, sizeof(unsigned));
			fin.read((char *)&planes, sizeof(unsigned short));
			if( planes != 1 )
			{
				cout << "Planes from " << fname << " is not 1: " << planes << "\n";
				return;
			}	
			fin.read((char *)&bpp, sizeof(unsigned short));	
			if( bpp != 24 )	{
				cout << "Bpp from " << fname << " is not 24: " << bpp << "\n";
				return;
			}
			fin.seekg(24, ios::cur);
			unsigned size(width * height * 3);				// size of the image in bytes (3 is to RGB component).	
			data = new unsigned char[size];	
			fin.read((char *)data, size);	
			unsigned char tmp;					// temporary color storage for bgr-rgb conversion.	
			for(unsigned int i(0); i < size; i += 3 )
			{	
				tmp = data[i];		
				data[i] = data[i+2];
				data[i+2] = tmp;
			}
		}

		~Bitmap()
		{
			if (data) free(data);
		}

		inline unsigned int getWidth() const { return width; }		// Gets width of image.
		inline unsigned int getHeight() const { return height; }	// Gets height of image.
		inline unsigned char* getData() const { return data; }		// Gets data stored in image (rgb format)
		inline unsigned short getPlanes() const { return planes; }	// Gets number of planes (1)
		inline unsigned short getBpp() const { return bpp; }		// Gets bits per pixels (24)
		inline bool isValid() const { return data != NULL; }		// Checks whether bmp was loaded successfully.
		
		// reads the r component wrapped at a certain position.
		inline unsigned char getR(int x, int y) const {
			if (!data) return 0;
			return data[ ((y%height) * width + x%width ) * 3];
		}

		// reads the g component wrapped at a certain position.
		inline unsigned char getG(int x, int y) const {
			if (!data) return 0;
			return data[ ((y%height) * width + x%width ) * 3 + 1];
		}

		// reads the b component wrapped at a certain position.
		inline unsigned char getB(int x, int y) const {
			if (!data) return 0;
			return data[ ((y%height) * width + x%width ) * 3 + 2];
		}

};

#endif