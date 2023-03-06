#include <Windows.h>
#include <stdio.h>

#include "System.h"

int _frame = 0;
int _width = 800;
int _height = 600;
float _zoom = -2.5f;
float _rotx = 0;
float _roty = 0.001f;
float _tx = 0;
float _ty = 0;
int _lastx = 0;
int _lasty = 0;
unsigned char _buttons[3] = { 0 };
bool _simulation = false;
char _FPS_str[100];

System* _system = nullptr;
Mesh* _mesh = nullptr;

#define SCREEN_CAPTURE

void DrawText(float x, float y, const char* text, void* font = NULL)
{
	glColor3f(0, 0, 0);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, (double)_width, 0.0, (double)_height, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	if (font == NULL) {
		font = GLUT_BITMAP_9_BY_15;
	}

	size_t len = strlen(text);

	glRasterPos2f(x, y);
	for (const char* letter = text; letter < text + len; letter++) {
		if (*letter == '\n') {
			y -= 12.0f;
			glRasterPos2f(x, y);
		}
		glutBitmapCharacter(font, *letter);
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
}
tuple<Mesh*, Mesh*> LoadClothAndAvatar(char *cloth_file, char *avatar_file, double scale)
{
	Mesh *cloth, *avatar;
	cloth = new Mesh(cloth_file);
	avatar = new Mesh(avatar_file);

	AABB aabb;
	setAABB(aabb, cloth->_aabb);
	addAABB(aabb, avatar->_aabb);

	REAL3 size = aabb._max - aabb._min;
	REAL max_length = size.x;
	if (max_length < size.y)
		max_length = size.y;
	if (max_length < size.z)
		max_length = size.z;
	max_length = 2.0 * scale / max_length;

	REAL3 prevCenter = (aabb._min + aabb._max) * (REAL)0.5;
	REAL3 center = make_REAL3(0.0, 0.0, 0.0);

	bool flag = false;
	uint vlen = cloth->_ns.size();
	for (uint i = 0u; i < vlen; i += 3u) {
		REAL3 pos = make_REAL3(cloth->_ns[i], cloth->_ns[i + 1u], cloth->_ns[i + 2u]);
		REAL3 grad = pos - prevCenter;
		grad *= max_length;
		pos = center + grad;
		cloth->_ns[i] = pos.x;
		cloth->_ns[i + 1u] = pos.y;
		cloth->_ns[i + 2u] = pos.z;
		if (flag) addAABB(aabb, pos);
		else {
			aabb._min = aabb._max = pos;
			flag = true;
		}
	}
	vlen = avatar->_ns.size();
	flag = false;
	for (uint i = 0u; i < vlen; i += 3u) {
		REAL3 pos = make_REAL3(avatar->_ns[i], avatar->_ns[i + 1u], avatar->_ns[i + 2u]);
		REAL3 grad = pos - prevCenter;
		grad *= max_length;
		pos = center + grad;
		avatar->_ns[i] = pos.x;
		avatar->_ns[i + 1u] = pos.y;
		avatar->_ns[i + 2u] = pos.z;
		if (flag) addAABB(aabb, pos);
		else {
			aabb._min = aabb._max = pos;
			flag = true;
		}
	}

	return make_tuple(cloth, avatar);
}

void SetView(double zoom, double tx, double ty, double rotx, double roty)
{
	_zoom = zoom;
	_tx = tx;
	_ty = ty;
	_rotx = rotx;
	_roty = roty;
}
void SetView(void)
{
	//----< Drop Boundary Collision >-----------------
	//SetView(-2.000000, 0.060000, 1.020000, 34.500000, 18.000999);
	//SetView(-3.899997, -1.020000, 0.480000, 4.500000, 183.001007);

	//----< Bunny Collision >-----------------
	//SetView(-1.700000, 0.000000, 0.960000, 36.000000, 56.500999);
	//SetView(-0.600000, -0.030000, 0.630000, 48.000000, -127.998993);
	
	//----< dragon Collision >-----------------
	//SetView(-1.750000, 0.030000, 0.840000, 31.000000, 15.500999);

	//----< Sphere Collision >-----------------
	SetView(-1.750000, 0.000000, 0.930000, 35.500000, 26.500999);
	//SetView(-1.100000, 0.000000, 1.289999, 0.000000, 0.001000);

	//----< Complex Collision >-----------------
	//SetView(-3.200000, -0.000000, 0.750000, 34.500000, 0.000000);
	//SetView(-2.500000, -0.000000, 1.100000, 34.000000, 0.000000);

	//----< Avatar Collision >-----------------
	//SetView(-0.950001, -0.000000, -0.390000, 22.000000, 33.500999);
	//SetView(-0.950001, -0.000000, -0.390000, 22.000000, 153.500999);

	//----< Stress Collision >-----------------
	//SetView(-2.500000, 0.000000, 0.000000, 7.000000, 0.000000);
}
void Init(void)
{
	//_system = new System(make_REAL3(0.0, -0.098, 0.0), 0.01);
	//_system = new System(make_REAL3(0.0, -0.0, 0.0), 0.01);
	_system = new System(make_REAL3(0.0, -0.98, 0.0), 0.01);
	//_system = new System(make_REAL3(0.0, -9.8, 0.0), 0.01);

	glEnable(GL_DEPTH_TEST);

	{
		////char* dir_cloth = "../obj/PD20_2.obj";
		////char* dir_cloth = "../obj/PD10_2.obj";
		//char* dir_cloth = "../obj/PD5_2.obj";

		////char* dir_avatar = "../obj/feifei_2.obj";
		//char* dir_avatar = "../obj/FeiFei_v2.obj";
		//
		//tuple<Mesh*, Mesh*> scene = LoadClothAndAvatar(dir_cloth, dir_avatar, 1.0);
		//_system->addCloth(get<0>(scene), 1.0);
		////_system->addObstacle(get<1>(scene), 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.8, 0.0));
		//_system->addObstacle(get<1>(scene), 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));
	}
	{
		//_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(0.0, 0.0, 0.0));
		////_mesh->rotate(make_REAL3(180.0, 0.0, 0.0));
		////_mesh->rotate(make_REAL3(0.0, 180.0, 0.0));
		//_system->addCloth(_mesh, 1.0);

		//_mesh = new Mesh("../obj/sphere.obj", make_REAL3(0.0, -10.0, 0.0), 0.1);
		//_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));
	}
	{
		/*_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(0.0, 0.0, 0.0));
		_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		_system->addCloth(_mesh, 1.0);

		_mesh = new Mesh("../obj/cube.obj", make_REAL3(0.0, -1.0, 0.0), 0.38);
		_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));*/
	}
	{
		_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(0.0, 0.0, 0.0));
		_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		_system->addCloth(_mesh, 1.0);

		_mesh = new Mesh("../obj/bunny.obj", make_REAL3(0.0, -1.0, 0.0), 0.4);
		//_mesh = new Mesh("../obj/dragon.obj", make_REAL3(0.0, -1.0, 0.0), 0.6);
		_system->addObstacle(_mesh, 1.0, make_REAL3(0.0, -1.0, 0.0), make_REAL3(0.0, 0.0, 0.0));
	}
	{
		/*_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(0.0, 0.0, 0.0));
		_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		_system->addCloth(_mesh, 1.0);

		_mesh = new Mesh("../obj/sphere.obj", make_REAL3(0.0, -1.0, 0.0), 0.18);
		_system->addObstacle(_mesh, 1.0, make_REAL3(0.0, -1.0, 0.0), make_REAL3(0.0, 1.0, 0.0));*/

		/*_mesh = new Mesh("../obj/sphere.obj", make_REAL3(0.0, -1.3, 0.0), 0.32);
		_system->addObstacle(_mesh, 1.0, make_REAL3(0.0, -1.3, 0.0), make_REAL3(0.0));*/
	}
	{
		//_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(-0.3, 0.0, -0.3), 1.0);
		//_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		//_system->addCloth(_mesh, 1.0);
		//delete _mesh;

		//_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(0.3, 0.02, -0.3), 1.0);
		//_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		//_system->addCloth(_mesh, 1.0);
		//delete _mesh;

		//_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(-0.3, 0.01, 0.3), 1.0);
		//_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		//_system->addCloth(_mesh, 1.0);
		//delete _mesh;

		//_mesh = new Mesh("../obj/HR_cloth.obj", make_REAL3(0.3, 0.03, 0.3), 1.0);
		//_mesh->rotate(make_REAL3(-90.0, 0.0, 0.0));
		//_system->addCloth(_mesh, 1.0);
		//delete _mesh;

		//_mesh = new Mesh("../obj/cube.obj", make_REAL3(0.75, -1.2, 0.75), 0.3);
		////_mesh = new Mesh("../obj/dragon.obj", make_REAL3(0.75, -1.2, 0.75), 0.5);
		//_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));
		//delete _mesh;

		//_mesh = new Mesh("../obj/cube.obj", make_REAL3(-0.75, -1.2, 0.75), 0.3);
		//_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));
		//delete _mesh;

		//_mesh = new Mesh("../obj/cube.obj", make_REAL3(0.75, -1.2, -0.75), 0.3);
		////_mesh = new Mesh("../obj/buddha.obj", make_REAL3(0.75, -1.2, -0.75), 0.5);
		//_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));
		//delete _mesh;

		//_mesh = new Mesh("../obj/cube.obj", make_REAL3(-0.75, -1.2, -0.75), 0.3);
		////_mesh = new Mesh("../obj/bunny2.obj", make_REAL3(-0.75, -1.2, -0.75), 0.5);
		//_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 0.0, 0.0));
		//delete _mesh;

		//_mesh = new Mesh("../obj/sphere.obj", make_REAL3(0.0, -0.9, 0.0), 0.28);
		////_mesh = new Mesh("../obj/cube.obj", make_REAL3(0.0, -1.0, 0.0), 0.23);
		//_system->addObstacle(_mesh, 1.0, make_REAL3(0.0), make_REAL3(0.0, 1.2, 0.0));
		//delete _mesh;
	}
}

void FPS(void)
{
	static float framesPerSecond = 0.0f;
	static float lastTime = 0.0f;
	float currentTime = GetTickCount() * 0.001f;
	++framesPerSecond;
	if (currentTime - lastTime > 1.0f) {
		lastTime = currentTime;
		sprintf(_FPS_str, "FPS : %d", (int)framesPerSecond);
		framesPerSecond = 0;
	}
}
void Darw(void)
{
	glutReshapeWindow(_width, _height);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glShadeModel(GL_SMOOTH);
	char text[100];

	if(_system)
		_system->draw();

	glDisable(GL_LIGHTING);
	//DrawText(10.0f, 580.0f, "Projective Dynamics, ACM TOG 2014");

	glDisable(GL_LIGHTING);
	if (_system)
		sprintf(text, "Number of triangles : %d", _system->_cloths->_numFaces);

	//DrawText(10.0f, 560.0f, text);
	DrawText(10.0f, 540.0f, _FPS_str);
	sprintf(text, "Frame : %d", _frame);
	DrawText(10.0f, 520.0f, text);
}
void Capture(int endFrame)
{
	if (_frame == 0 || _frame % 2 == 0) {
		static int index = 0;
		char filename[100];
		sprintf_s(filename, "../capture/capture-%d.bmp", index);
		BITMAPFILEHEADER bf;
		BITMAPINFOHEADER bi;
		unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * _width * _height * 3);
		FILE* file;
		fopen_s(&file, filename, "wb");
		if (image != NULL) {
			if (file != NULL) {
				glReadPixels(0, 0, _width, _height, 0x80E0, GL_UNSIGNED_BYTE, image);
				memset(&bf, 0, sizeof(bf));
				memset(&bi, 0, sizeof(bi));
				bf.bfType = 'MB';
				bf.bfSize = sizeof(bf) + sizeof(bi) + _width * _height * 3;
				bf.bfOffBits = sizeof(bf) + sizeof(bi);
				bi.biSize = sizeof(bi);
				bi.biWidth = _width;
				bi.biHeight = _height;
				bi.biPlanes = 1;
				bi.biBitCount = 24;
				bi.biSizeImage = _width * _height * 3;
				fwrite(&bf, sizeof(bf), 1, file);
				fwrite(&bi, sizeof(bi), 1, file);
				fwrite(image, sizeof(unsigned char), _height * _width * 3, file);
				fclose(file);
			}
			free(image);
		}
		if (index == endFrame) {
			exit(0);
		}
		index++;
	}
}
void Update(void)
{
	if (_simulation) {
#ifdef SCREEN_CAPTURE
		Capture(2050);
#endif
		if (_system)
			_system->simulation();
		_frame++;
	}
	::glutPostRedisplay();
}

void Display(void)
{
	glClearColor(0.8980392156862745f, 0.9490196078431373f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.1f, 4.0f);
	glLoadIdentity();

	glTranslatef(0, 0, _zoom);
	glTranslatef(_tx, _ty, 0);
	glRotatef(_rotx, 1, 0, 0);
	glRotatef(_roty, 0, 1, 0);

	SetView();

	//glTranslatef(-0.5f, -0.5f, -0.5f);
	Darw();
	FPS();
	glutSwapBuffers();
}

void Reshape(int w, int h)
{
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (float)w / h, 0.1, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Motion(int x, int y)
{
	int diffx = x - _lastx;
	int diffy = y - _lasty;
	_lastx = x;
	_lasty = y;

	if (_buttons[2]) {
		_zoom += (float)0.05f * diffx;
	}
	else if (_buttons[0]) {
		_rotx += (float)0.5f * diffy;
		_roty += (float)0.5f * diffx;
	}
	else if (_buttons[1]) {
		_tx += (float)0.03f * diffx;
		_ty -= (float)0.03f * diffy;
	}

	if (_simulation) {
		SetView();
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
	_lastx = x;
	_lasty = y;
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		_buttons[0] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_MIDDLE_BUTTON:
		_buttons[1] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_RIGHT_BUTTON:
		_buttons[2] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void SpecialInput(int key, int x, int y)
{
	glutPostRedisplay();
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
	case 'Q':
		exit(0);
	case ' ':
		_simulation = !_simulation;
		break;
	case 'r':
	case 'R':
		_system->reset();
		break;
	case 'c':
	case 'C':
		printf("%f, %f, %f, %f, %f\n", _zoom, _tx, _ty, _rotx, _roty);
		break;
	case 'd':
	case 'D':
		//_system->_bvh->_test++;
		//CollisionSolver::Debug();
		break;
	}
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	cudaDeviceReset();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(_width, _height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Cloth Simulator");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Update);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialInput);
	Init();
	glutMainLoop();
}