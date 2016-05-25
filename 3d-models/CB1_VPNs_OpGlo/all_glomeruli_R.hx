# Amira Script
remove -all
remove CB1_binary.surf LC04_opGlo_l_VT031479.surf LC06_opGlo_l_VT009855.surf LC09_opGlo_l_VT014209.surf LC10_opGlo_l_VT021760.surf LC11_opGlo_l_VT004968.surf LC12_opGlo_l_VT040919.surf LC13_opGlo_l_GMR50C10.surf LC15_opGlo_l_VT014207.surf LC16_opGlo_l_VT061079.surf all_glomeruli_R.surf LC18_opGlo_l_GMR92B11.surf LC20_opGlo_l_VT025718.surf LC21_opGlo_l_GMR85F11.surf LC22_LPLC4_opGlo_l_VT058688.surf LC24_opGlo_l_VT038216.surf LPC1_opGlo_l_GMR77A06.surf LPLC1_opGlo_l_GMR36B06.surf LPLC2_opGlo_l_VT007194.surf LPLC3_opGlo_l_VT044492.surf RadiossSurfaceView RadiossSurfaceView2 RadiossSurfaceView3 RadiossSurfaceView4 RadiossSurfaceView5 RadiossSurfaceView6 RadiossSurfaceView7 RadiossSurfaceView8 RadiossSurfaceView9 RadiossSurfaceView10 RadiossSurfaceView11 RadiossSurfaceView12 RadiossSurfaceView13 RadiossSurfaceView14 RadiossSurfaceView15 RadiossSurfaceView16 RadiossSurfaceView17 RadiossSurfaceView18 RadiossSurfaceView19 RadiossSurfaceView20

# Create viewers
viewer setVertical 0

viewer 0 setBackgroundMode 0
viewer 0 setBackgroundColor 1 1 1
viewer 0 setBackgroundColor2 0.72 0.72 0.78
viewer 0 setTransparencyType 5
viewer 0 setAutoRedraw 0
viewer 0 setCameraType 1
viewer 0 show
mainWindow show

set hideNewModules 0
[ load U:/GLomeruliCLustering_final/Figures/FIGURES_DATA/DATA/Template_Brains_(MultipleBrainAveraged)/CB1/CB1_binary.surf ] setLabel CB1_binary.surf
CB1_binary.surf setIconPosition 20 10
CB1_binary.surf fire
CB1_binary.surf LevelOfDetail setMinMax -1 -1
CB1_binary.surf LevelOfDetail setButtons 1
CB1_binary.surf LevelOfDetail setIncrement 1
CB1_binary.surf LevelOfDetail setValue -1
CB1_binary.surf LevelOfDetail setSubMinMax -1 -1
CB1_binary.surf fire
CB1_binary.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC04_opGlo_l_VT031479.surf ] setLabel LC04_opGlo_l_VT031479.surf
LC04_opGlo_l_VT031479.surf setIconPosition 20 40
LC04_opGlo_l_VT031479.surf fire
LC04_opGlo_l_VT031479.surf LevelOfDetail setMinMax -1 -1
LC04_opGlo_l_VT031479.surf LevelOfDetail setButtons 1
LC04_opGlo_l_VT031479.surf LevelOfDetail setIncrement 1
LC04_opGlo_l_VT031479.surf LevelOfDetail setValue -1
LC04_opGlo_l_VT031479.surf LevelOfDetail setSubMinMax -1 -1
LC04_opGlo_l_VT031479.surf fire
LC04_opGlo_l_VT031479.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LC06_opGlo_l_VT009855.surf ] setLabel LC06_opGlo_l_VT009855.surf
LC06_opGlo_l_VT009855.surf setIconPosition 20 70
LC06_opGlo_l_VT009855.surf fire
LC06_opGlo_l_VT009855.surf LevelOfDetail setMinMax -1 -1
LC06_opGlo_l_VT009855.surf LevelOfDetail setButtons 1
LC06_opGlo_l_VT009855.surf LevelOfDetail setIncrement 1
LC06_opGlo_l_VT009855.surf LevelOfDetail setValue -1
LC06_opGlo_l_VT009855.surf LevelOfDetail setSubMinMax -1 -1
LC06_opGlo_l_VT009855.surf fire
LC06_opGlo_l_VT009855.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC09_opGlo_l_VT014209.surf ] setLabel LC09_opGlo_l_VT014209.surf
LC09_opGlo_l_VT014209.surf setIconPosition 20 100
LC09_opGlo_l_VT014209.surf fire
LC09_opGlo_l_VT014209.surf LevelOfDetail setMinMax -1 -1
LC09_opGlo_l_VT014209.surf LevelOfDetail setButtons 1
LC09_opGlo_l_VT014209.surf LevelOfDetail setIncrement 1
LC09_opGlo_l_VT014209.surf LevelOfDetail setValue -1
LC09_opGlo_l_VT014209.surf LevelOfDetail setSubMinMax -1 -1
LC09_opGlo_l_VT014209.surf fire
LC09_opGlo_l_VT014209.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC10_opGlo_l_VT021760.surf ] setLabel LC10_opGlo_l_VT021760.surf
LC10_opGlo_l_VT021760.surf setIconPosition 20 130
LC10_opGlo_l_VT021760.surf fire
LC10_opGlo_l_VT021760.surf LevelOfDetail setMinMax -1 -1
LC10_opGlo_l_VT021760.surf LevelOfDetail setButtons 1
LC10_opGlo_l_VT021760.surf LevelOfDetail setIncrement 1
LC10_opGlo_l_VT021760.surf LevelOfDetail setValue -1
LC10_opGlo_l_VT021760.surf LevelOfDetail setSubMinMax -1 -1
LC10_opGlo_l_VT021760.surf fire
LC10_opGlo_l_VT021760.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC11_opGlo_l_VT004968.surf ] setLabel LC11_opGlo_l_VT004968.surf
LC11_opGlo_l_VT004968.surf setIconPosition 19 161
LC11_opGlo_l_VT004968.surf fire
LC11_opGlo_l_VT004968.surf LevelOfDetail setMinMax -1 -1
LC11_opGlo_l_VT004968.surf LevelOfDetail setButtons 1
LC11_opGlo_l_VT004968.surf LevelOfDetail setIncrement 1
LC11_opGlo_l_VT004968.surf LevelOfDetail setValue -1
LC11_opGlo_l_VT004968.surf LevelOfDetail setSubMinMax -1 -1
LC11_opGlo_l_VT004968.surf fire
LC11_opGlo_l_VT004968.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC12_opGlo_l_VT040919.surf ] setLabel LC12_opGlo_l_VT040919.surf
LC12_opGlo_l_VT040919.surf setIconPosition 20 190
LC12_opGlo_l_VT040919.surf fire
LC12_opGlo_l_VT040919.surf LevelOfDetail setMinMax -1 -1
LC12_opGlo_l_VT040919.surf LevelOfDetail setButtons 1
LC12_opGlo_l_VT040919.surf LevelOfDetail setIncrement 1
LC12_opGlo_l_VT040919.surf LevelOfDetail setValue -1
LC12_opGlo_l_VT040919.surf LevelOfDetail setSubMinMax -1 -1
LC12_opGlo_l_VT040919.surf fire
LC12_opGlo_l_VT040919.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC13_opGlo_l_GMR50C10.surf ] setLabel LC13_opGlo_l_GMR50C10.surf
LC13_opGlo_l_GMR50C10.surf setIconPosition 20 220
LC13_opGlo_l_GMR50C10.surf fire
LC13_opGlo_l_GMR50C10.surf LevelOfDetail setMinMax -1 -1
LC13_opGlo_l_GMR50C10.surf LevelOfDetail setButtons 1
LC13_opGlo_l_GMR50C10.surf LevelOfDetail setIncrement 1
LC13_opGlo_l_GMR50C10.surf LevelOfDetail setValue -1
LC13_opGlo_l_GMR50C10.surf LevelOfDetail setSubMinMax -1 -1
LC13_opGlo_l_GMR50C10.surf fire
LC13_opGlo_l_GMR50C10.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC15_opGlo_l_VT014207.surf ] setLabel LC15_opGlo_l_VT014207.surf
LC15_opGlo_l_VT014207.surf setIconPosition 20 250
LC15_opGlo_l_VT014207.surf fire
LC15_opGlo_l_VT014207.surf LevelOfDetail setMinMax -1 -1
LC15_opGlo_l_VT014207.surf LevelOfDetail setButtons 1
LC15_opGlo_l_VT014207.surf LevelOfDetail setIncrement 1
LC15_opGlo_l_VT014207.surf LevelOfDetail setValue -1
LC15_opGlo_l_VT014207.surf LevelOfDetail setSubMinMax -1 -1
LC15_opGlo_l_VT014207.surf fire
LC15_opGlo_l_VT014207.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC16_opGlo_l_VT061079.surf ] setLabel LC16_opGlo_l_VT061079.surf
LC16_opGlo_l_VT061079.surf setIconPosition 20 280
LC16_opGlo_l_VT061079.surf fire
LC16_opGlo_l_VT061079.surf LevelOfDetail setMinMax -1 -1
LC16_opGlo_l_VT061079.surf LevelOfDetail setButtons 1
LC16_opGlo_l_VT061079.surf LevelOfDetail setIncrement 1
LC16_opGlo_l_VT061079.surf LevelOfDetail setValue -1
LC16_opGlo_l_VT061079.surf LevelOfDetail setSubMinMax -1 -1
LC16_opGlo_l_VT061079.surf fire
LC16_opGlo_l_VT061079.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R.surf ] setLabel all_glomeruli_R.surf
all_glomeruli_R.surf setIconPosition 20 310
all_glomeruli_R.surf fire
all_glomeruli_R.surf LevelOfDetail setMinMax -1 -1
all_glomeruli_R.surf LevelOfDetail setButtons 1
all_glomeruli_R.surf LevelOfDetail setIncrement 1
all_glomeruli_R.surf LevelOfDetail setValue -1
all_glomeruli_R.surf LevelOfDetail setSubMinMax -1 -1
all_glomeruli_R.surf fire
all_glomeruli_R.surf setViewerMask 65535
all_glomeruli_R.surf select

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC18_opGlo_l_GMR92B11.surf ] setLabel LC18_opGlo_l_GMR92B11.surf
LC18_opGlo_l_GMR92B11.surf setIconPosition 20 340
LC18_opGlo_l_GMR92B11.surf fire
LC18_opGlo_l_GMR92B11.surf LevelOfDetail setMinMax -1 -1
LC18_opGlo_l_GMR92B11.surf LevelOfDetail setButtons 1
LC18_opGlo_l_GMR92B11.surf LevelOfDetail setIncrement 1
LC18_opGlo_l_GMR92B11.surf LevelOfDetail setValue -1
LC18_opGlo_l_GMR92B11.surf LevelOfDetail setSubMinMax -1 -1
LC18_opGlo_l_GMR92B11.surf fire
LC18_opGlo_l_GMR92B11.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LC20_opGlo_l_VT025718.surf ] setLabel LC20_opGlo_l_VT025718.surf
LC20_opGlo_l_VT025718.surf setIconPosition 20 370
LC20_opGlo_l_VT025718.surf fire
LC20_opGlo_l_VT025718.surf LevelOfDetail setMinMax -1 -1
LC20_opGlo_l_VT025718.surf LevelOfDetail setButtons 1
LC20_opGlo_l_VT025718.surf LevelOfDetail setIncrement 1
LC20_opGlo_l_VT025718.surf LevelOfDetail setValue -1
LC20_opGlo_l_VT025718.surf LevelOfDetail setSubMinMax -1 -1
LC20_opGlo_l_VT025718.surf fire
LC20_opGlo_l_VT025718.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC21_opGlo_l_GMR85F11.surf ] setLabel LC21_opGlo_l_GMR85F11.surf
LC21_opGlo_l_GMR85F11.surf setIconPosition 20 400
LC21_opGlo_l_GMR85F11.surf fire
LC21_opGlo_l_GMR85F11.surf LevelOfDetail setMinMax -1 -1
LC21_opGlo_l_GMR85F11.surf LevelOfDetail setButtons 1
LC21_opGlo_l_GMR85F11.surf LevelOfDetail setIncrement 1
LC21_opGlo_l_GMR85F11.surf LevelOfDetail setValue -1
LC21_opGlo_l_GMR85F11.surf LevelOfDetail setSubMinMax -1 -1
LC21_opGlo_l_GMR85F11.surf fire
LC21_opGlo_l_GMR85F11.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LC22_LPLC4_opGlo_l_VT058688.surf ] setLabel LC22_LPLC4_opGlo_l_VT058688.surf
LC22_LPLC4_opGlo_l_VT058688.surf setIconPosition 20 430
LC22_LPLC4_opGlo_l_VT058688.surf fire
LC22_LPLC4_opGlo_l_VT058688.surf LevelOfDetail setMinMax -1 -1
LC22_LPLC4_opGlo_l_VT058688.surf LevelOfDetail setButtons 1
LC22_LPLC4_opGlo_l_VT058688.surf LevelOfDetail setIncrement 1
LC22_LPLC4_opGlo_l_VT058688.surf LevelOfDetail setValue -1
LC22_LPLC4_opGlo_l_VT058688.surf LevelOfDetail setSubMinMax -1 -1
LC22_LPLC4_opGlo_l_VT058688.surf fire
LC22_LPLC4_opGlo_l_VT058688.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/all_glomeruli_R-files/LC24_opGlo_l_VT038216.surf ] setLabel LC24_opGlo_l_VT038216.surf
LC24_opGlo_l_VT038216.surf setIconPosition 20 460
LC24_opGlo_l_VT038216.surf fire
LC24_opGlo_l_VT038216.surf LevelOfDetail setMinMax -1 -1
LC24_opGlo_l_VT038216.surf LevelOfDetail setButtons 1
LC24_opGlo_l_VT038216.surf LevelOfDetail setIncrement 1
LC24_opGlo_l_VT038216.surf LevelOfDetail setValue -1
LC24_opGlo_l_VT038216.surf LevelOfDetail setSubMinMax -1 -1
LC24_opGlo_l_VT038216.surf fire
LC24_opGlo_l_VT038216.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LPC1_opGlo_l_GMR77A06.surf ] setLabel LPC1_opGlo_l_GMR77A06.surf
LPC1_opGlo_l_GMR77A06.surf setIconPosition 20 490
LPC1_opGlo_l_GMR77A06.surf fire
LPC1_opGlo_l_GMR77A06.surf LevelOfDetail setMinMax -1 -1
LPC1_opGlo_l_GMR77A06.surf LevelOfDetail setButtons 1
LPC1_opGlo_l_GMR77A06.surf LevelOfDetail setIncrement 1
LPC1_opGlo_l_GMR77A06.surf LevelOfDetail setValue -1
LPC1_opGlo_l_GMR77A06.surf LevelOfDetail setSubMinMax -1 -1
LPC1_opGlo_l_GMR77A06.surf fire
LPC1_opGlo_l_GMR77A06.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LPLC1_opGlo_l_GMR36B06.surf ] setLabel LPLC1_opGlo_l_GMR36B06.surf
LPLC1_opGlo_l_GMR36B06.surf setIconPosition 20 520
LPLC1_opGlo_l_GMR36B06.surf fire
LPLC1_opGlo_l_GMR36B06.surf LevelOfDetail setMinMax -1 -1
LPLC1_opGlo_l_GMR36B06.surf LevelOfDetail setButtons 1
LPLC1_opGlo_l_GMR36B06.surf LevelOfDetail setIncrement 1
LPLC1_opGlo_l_GMR36B06.surf LevelOfDetail setValue -1
LPLC1_opGlo_l_GMR36B06.surf LevelOfDetail setSubMinMax -1 -1
LPLC1_opGlo_l_GMR36B06.surf fire
LPLC1_opGlo_l_GMR36B06.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LPLC2_opGlo_l_VT007194.surf ] setLabel LPLC2_opGlo_l_VT007194.surf
LPLC2_opGlo_l_VT007194.surf setIconPosition 20 550
LPLC2_opGlo_l_VT007194.surf fire
LPLC2_opGlo_l_VT007194.surf LevelOfDetail setMinMax -1 -1
LPLC2_opGlo_l_VT007194.surf LevelOfDetail setButtons 1
LPLC2_opGlo_l_VT007194.surf LevelOfDetail setIncrement 1
LPLC2_opGlo_l_VT007194.surf LevelOfDetail setValue -1
LPLC2_opGlo_l_VT007194.surf LevelOfDetail setSubMinMax -1 -1
LPLC2_opGlo_l_VT007194.surf fire
LPLC2_opGlo_l_VT007194.surf setViewerMask 65535

set hideNewModules 0
[ load ${SCRIPTDIR}/Segmentations_left/LPLC3_opGlo_l_VT044492.surf ] setLabel LPLC3_opGlo_l_VT044492.surf
LPLC3_opGlo_l_VT044492.surf setIconPosition 20 580
LPLC3_opGlo_l_VT044492.surf fire
LPLC3_opGlo_l_VT044492.surf LevelOfDetail setMinMax -1 -1
LPLC3_opGlo_l_VT044492.surf LevelOfDetail setButtons 1
LPLC3_opGlo_l_VT044492.surf LevelOfDetail setIncrement 1
LPLC3_opGlo_l_VT044492.surf LevelOfDetail setValue -1
LPLC3_opGlo_l_VT044492.surf LevelOfDetail setSubMinMax -1 -1
LPLC3_opGlo_l_VT044492.surf fire
LPLC3_opGlo_l_VT044492.surf setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView}
RadiossSurfaceView setIconPosition 385 10
RadiossSurfaceView data connect CB1_binary.surf
RadiossSurfaceView colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView colormap setDefaultAlpha 0.500000
RadiossSurfaceView fire
RadiossSurfaceView drawStyle setValue 4
RadiossSurfaceView drawStyle setSpecularLighting 1
RadiossSurfaceView drawStyle setTexture 0
RadiossSurfaceView drawStyle setAlphaMode 3
RadiossSurfaceView drawStyle setNormalBinding 0
RadiossSurfaceView drawStyle setCullingMode 0
RadiossSurfaceView drawStyle setSortingMode 1
RadiossSurfaceView selectionMode setValue 0 0
RadiossSurfaceView Patch setMinMax 0 1
RadiossSurfaceView Patch setButtons 1
RadiossSurfaceView Patch setIncrement 1
RadiossSurfaceView Patch setValue 0
RadiossSurfaceView Patch setSubMinMax 0 1
RadiossSurfaceView BoundaryId setValue 0 -1
RadiossSurfaceView materials setValue 0 1
RadiossSurfaceView materials setValue 1 0
RadiossSurfaceView colorMode setValue 0
RadiossSurfaceView baseTrans setMinMax 0 1
RadiossSurfaceView baseTrans setButtons 0
RadiossSurfaceView baseTrans setIncrement 0.1
RadiossSurfaceView baseTrans setValue 0.893617
RadiossSurfaceView baseTrans setSubMinMax 0 1
RadiossSurfaceView VRMode setValue 0 0
RadiossSurfaceView fire
RadiossSurfaceView hideBox 1
{RadiossSurfaceView} selectTriangles zab HIJMONMDABANAAAAAMACKAPHCPONENGAACNIEIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGALLHOMMJMPKLI
RadiossSurfaceView fire
RadiossSurfaceView setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView2}
RadiossSurfaceView2 setIconPosition 379 40
RadiossSurfaceView2 data connect LC04_opGlo_l_VT031479.surf
RadiossSurfaceView2 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView2 colormap setDefaultAlpha 0.500000
RadiossSurfaceView2 fire
RadiossSurfaceView2 drawStyle setValue 1
RadiossSurfaceView2 drawStyle setSpecularLighting 1
RadiossSurfaceView2 drawStyle setTexture 0
RadiossSurfaceView2 drawStyle setAlphaMode 1
RadiossSurfaceView2 drawStyle setNormalBinding 0
RadiossSurfaceView2 drawStyle setCullingMode 0
RadiossSurfaceView2 drawStyle setSortingMode 1
RadiossSurfaceView2 selectionMode setValue 0 0
RadiossSurfaceView2 Patch setMinMax 0 1
RadiossSurfaceView2 Patch setButtons 1
RadiossSurfaceView2 Patch setIncrement 1
RadiossSurfaceView2 Patch setValue 0
RadiossSurfaceView2 Patch setSubMinMax 0 1
RadiossSurfaceView2 BoundaryId setValue 0 -1
RadiossSurfaceView2 materials setValue 0 1
RadiossSurfaceView2 materials setValue 1 0
RadiossSurfaceView2 colorMode setValue 0
RadiossSurfaceView2 baseTrans setMinMax 0 1
RadiossSurfaceView2 baseTrans setButtons 0
RadiossSurfaceView2 baseTrans setIncrement 0.1
RadiossSurfaceView2 baseTrans setValue 0.8
RadiossSurfaceView2 baseTrans setSubMinMax 0 1
RadiossSurfaceView2 VRMode setValue 0 0
RadiossSurfaceView2 fire
RadiossSurfaceView2 hideBox 1
{RadiossSurfaceView2} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBCICABODAPAPPPPAPAANFNLDEDA
RadiossSurfaceView2 fire
RadiossSurfaceView2 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView3}
RadiossSurfaceView3 setIconPosition 379 70
RadiossSurfaceView3 data connect LC06_opGlo_l_VT009855.surf
RadiossSurfaceView3 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView3 colormap setDefaultAlpha 0.500000
RadiossSurfaceView3 fire
RadiossSurfaceView3 drawStyle setValue 1
RadiossSurfaceView3 drawStyle setSpecularLighting 1
RadiossSurfaceView3 drawStyle setTexture 0
RadiossSurfaceView3 drawStyle setAlphaMode 1
RadiossSurfaceView3 drawStyle setNormalBinding 0
RadiossSurfaceView3 drawStyle setCullingMode 0
RadiossSurfaceView3 drawStyle setSortingMode 1
RadiossSurfaceView3 selectionMode setValue 0 0
RadiossSurfaceView3 Patch setMinMax 0 1
RadiossSurfaceView3 Patch setButtons 1
RadiossSurfaceView3 Patch setIncrement 1
RadiossSurfaceView3 Patch setValue 0
RadiossSurfaceView3 Patch setSubMinMax 0 1
RadiossSurfaceView3 BoundaryId setValue 0 -1
RadiossSurfaceView3 materials setValue 0 1
RadiossSurfaceView3 materials setValue 1 0
RadiossSurfaceView3 colorMode setValue 0
RadiossSurfaceView3 baseTrans setMinMax 0 1
RadiossSurfaceView3 baseTrans setButtons 0
RadiossSurfaceView3 baseTrans setIncrement 0.1
RadiossSurfaceView3 baseTrans setValue 0.8
RadiossSurfaceView3 baseTrans setSubMinMax 0 1
RadiossSurfaceView3 VRMode setValue 0 0
RadiossSurfaceView3 fire
RadiossSurfaceView3 hideBox 1
{RadiossSurfaceView3} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBEAAAHOCAAGAAHGHAHNNI
RadiossSurfaceView3 fire
RadiossSurfaceView3 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView4}
RadiossSurfaceView4 setIconPosition 379 100
RadiossSurfaceView4 data connect LC09_opGlo_l_VT014209.surf
RadiossSurfaceView4 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView4 colormap setDefaultAlpha 0.500000
RadiossSurfaceView4 fire
RadiossSurfaceView4 drawStyle setValue 1
RadiossSurfaceView4 drawStyle setSpecularLighting 1
RadiossSurfaceView4 drawStyle setTexture 0
RadiossSurfaceView4 drawStyle setAlphaMode 1
RadiossSurfaceView4 drawStyle setNormalBinding 0
RadiossSurfaceView4 drawStyle setCullingMode 0
RadiossSurfaceView4 drawStyle setSortingMode 1
RadiossSurfaceView4 selectionMode setValue 0 0
RadiossSurfaceView4 Patch setMinMax 0 1
RadiossSurfaceView4 Patch setButtons 1
RadiossSurfaceView4 Patch setIncrement 1
RadiossSurfaceView4 Patch setValue 0
RadiossSurfaceView4 Patch setSubMinMax 0 1
RadiossSurfaceView4 BoundaryId setValue 0 -1
RadiossSurfaceView4 materials setValue 0 1
RadiossSurfaceView4 materials setValue 1 0
RadiossSurfaceView4 colorMode setValue 0
RadiossSurfaceView4 baseTrans setMinMax 0 1
RadiossSurfaceView4 baseTrans setButtons 0
RadiossSurfaceView4 baseTrans setIncrement 0.1
RadiossSurfaceView4 baseTrans setValue 0.8
RadiossSurfaceView4 baseTrans setSubMinMax 0 1
RadiossSurfaceView4 VRMode setValue 0 0
RadiossSurfaceView4 fire
RadiossSurfaceView4 hideBox 1
{RadiossSurfaceView4} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBCIBIAFKDIADGIAABIIABJJKJGDAO
RadiossSurfaceView4 fire
RadiossSurfaceView4 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView5}
RadiossSurfaceView5 setIconPosition 379 130
RadiossSurfaceView5 data connect LC10_opGlo_l_VT021760.surf
RadiossSurfaceView5 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView5 colormap setDefaultAlpha 0.500000
RadiossSurfaceView5 fire
RadiossSurfaceView5 drawStyle setValue 1
RadiossSurfaceView5 drawStyle setSpecularLighting 1
RadiossSurfaceView5 drawStyle setTexture 0
RadiossSurfaceView5 drawStyle setAlphaMode 1
RadiossSurfaceView5 drawStyle setNormalBinding 0
RadiossSurfaceView5 drawStyle setCullingMode 0
RadiossSurfaceView5 drawStyle setSortingMode 1
RadiossSurfaceView5 selectionMode setValue 0 0
RadiossSurfaceView5 Patch setMinMax 0 1
RadiossSurfaceView5 Patch setButtons 1
RadiossSurfaceView5 Patch setIncrement 1
RadiossSurfaceView5 Patch setValue 0
RadiossSurfaceView5 Patch setSubMinMax 0 1
RadiossSurfaceView5 BoundaryId setValue 0 -1
RadiossSurfaceView5 materials setValue 0 1
RadiossSurfaceView5 materials setValue 1 0
RadiossSurfaceView5 colorMode setValue 0
RadiossSurfaceView5 baseTrans setMinMax 0 1
RadiossSurfaceView5 baseTrans setButtons 0
RadiossSurfaceView5 baseTrans setIncrement 0.1
RadiossSurfaceView5 baseTrans setValue 0.8
RadiossSurfaceView5 baseTrans setSubMinMax 0 1
RadiossSurfaceView5 VRMode setValue 0 0
RadiossSurfaceView5 fire
RadiossSurfaceView5 hideBox 1
{RadiossSurfaceView5} selectTriangles zab HIJMONMBABABAAAAAIMDKAPHCPDNIDAIBEAAAANPGNOLAADAIPALIC
RadiossSurfaceView5 fire
RadiossSurfaceView5 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView6}
RadiossSurfaceView6 setIconPosition 379 160
RadiossSurfaceView6 data connect LC11_opGlo_l_VT004968.surf
RadiossSurfaceView6 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView6 colormap setDefaultAlpha 0.500000
RadiossSurfaceView6 fire
RadiossSurfaceView6 drawStyle setValue 1
RadiossSurfaceView6 drawStyle setSpecularLighting 1
RadiossSurfaceView6 drawStyle setTexture 0
RadiossSurfaceView6 drawStyle setAlphaMode 1
RadiossSurfaceView6 drawStyle setNormalBinding 0
RadiossSurfaceView6 drawStyle setCullingMode 0
RadiossSurfaceView6 drawStyle setSortingMode 1
RadiossSurfaceView6 selectionMode setValue 0 0
RadiossSurfaceView6 Patch setMinMax 0 1
RadiossSurfaceView6 Patch setButtons 1
RadiossSurfaceView6 Patch setIncrement 1
RadiossSurfaceView6 Patch setValue 0
RadiossSurfaceView6 Patch setSubMinMax 0 1
RadiossSurfaceView6 BoundaryId setValue 0 -1
RadiossSurfaceView6 materials setValue 0 1
RadiossSurfaceView6 materials setValue 1 0
RadiossSurfaceView6 colorMode setValue 0
RadiossSurfaceView6 baseTrans setMinMax 0 1
RadiossSurfaceView6 baseTrans setButtons 0
RadiossSurfaceView6 baseTrans setIncrement 0.1
RadiossSurfaceView6 baseTrans setValue 0.8
RadiossSurfaceView6 baseTrans setSubMinMax 0 1
RadiossSurfaceView6 VRMode setValue 0 0
RadiossSurfaceView6 fire
RadiossSurfaceView6 hideBox 1
{RadiossSurfaceView6} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBCIBIAFEDANDAAADBAAIFLALLKH
RadiossSurfaceView6 fire
RadiossSurfaceView6 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView7}
RadiossSurfaceView7 setIconPosition 379 190
RadiossSurfaceView7 data connect LC12_opGlo_l_VT040919.surf
RadiossSurfaceView7 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView7 colormap setDefaultAlpha 0.500000
RadiossSurfaceView7 fire
RadiossSurfaceView7 drawStyle setValue 1
RadiossSurfaceView7 drawStyle setSpecularLighting 1
RadiossSurfaceView7 drawStyle setTexture 0
RadiossSurfaceView7 drawStyle setAlphaMode 1
RadiossSurfaceView7 drawStyle setNormalBinding 0
RadiossSurfaceView7 drawStyle setCullingMode 0
RadiossSurfaceView7 drawStyle setSortingMode 1
RadiossSurfaceView7 selectionMode setValue 0 0
RadiossSurfaceView7 Patch setMinMax 0 1
RadiossSurfaceView7 Patch setButtons 1
RadiossSurfaceView7 Patch setIncrement 1
RadiossSurfaceView7 Patch setValue 0
RadiossSurfaceView7 Patch setSubMinMax 0 1
RadiossSurfaceView7 BoundaryId setValue 0 -1
RadiossSurfaceView7 materials setValue 0 1
RadiossSurfaceView7 materials setValue 1 0
RadiossSurfaceView7 colorMode setValue 0
RadiossSurfaceView7 baseTrans setMinMax 0 1
RadiossSurfaceView7 baseTrans setButtons 0
RadiossSurfaceView7 baseTrans setIncrement 0.1
RadiossSurfaceView7 baseTrans setValue 0.8
RadiossSurfaceView7 baseTrans setSubMinMax 0 1
RadiossSurfaceView7 VRMode setValue 0 0
RadiossSurfaceView7 fire
RadiossSurfaceView7 hideBox 1
{RadiossSurfaceView7} selectTriangles zab HIJMONMBIBAAAAAAAIMALAAMPCKHHNACCBGMCLAAAADOLDHFMANGCBGN
RadiossSurfaceView7 fire
RadiossSurfaceView7 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView8}
RadiossSurfaceView8 setIconPosition 379 220
RadiossSurfaceView8 data connect LC13_opGlo_l_GMR50C10.surf
RadiossSurfaceView8 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView8 colormap setDefaultAlpha 0.500000
RadiossSurfaceView8 fire
RadiossSurfaceView8 drawStyle setValue 1
RadiossSurfaceView8 drawStyle setSpecularLighting 1
RadiossSurfaceView8 drawStyle setTexture 0
RadiossSurfaceView8 drawStyle setAlphaMode 1
RadiossSurfaceView8 drawStyle setNormalBinding 0
RadiossSurfaceView8 drawStyle setCullingMode 0
RadiossSurfaceView8 drawStyle setSortingMode 1
RadiossSurfaceView8 selectionMode setValue 0 0
RadiossSurfaceView8 Patch setMinMax 0 1
RadiossSurfaceView8 Patch setButtons 1
RadiossSurfaceView8 Patch setIncrement 1
RadiossSurfaceView8 Patch setValue 0
RadiossSurfaceView8 Patch setSubMinMax 0 1
RadiossSurfaceView8 BoundaryId setValue 0 -1
RadiossSurfaceView8 materials setValue 0 1
RadiossSurfaceView8 materials setValue 1 0
RadiossSurfaceView8 colorMode setValue 0
RadiossSurfaceView8 baseTrans setMinMax 0 1
RadiossSurfaceView8 baseTrans setButtons 0
RadiossSurfaceView8 baseTrans setIncrement 0.1
RadiossSurfaceView8 baseTrans setValue 0.8
RadiossSurfaceView8 baseTrans setSubMinMax 0 1
RadiossSurfaceView8 VRMode setValue 0 0
RadiossSurfaceView8 fire
RadiossSurfaceView8 hideBox 1
{RadiossSurfaceView8} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGABCNAAADADMDHPAAIFHHFLOL
RadiossSurfaceView8 fire
RadiossSurfaceView8 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView9}
RadiossSurfaceView9 setIconPosition 379 250
RadiossSurfaceView9 data connect LC15_opGlo_l_VT014207.surf
RadiossSurfaceView9 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView9 colormap setDefaultAlpha 0.500000
RadiossSurfaceView9 fire
RadiossSurfaceView9 drawStyle setValue 1
RadiossSurfaceView9 drawStyle setSpecularLighting 1
RadiossSurfaceView9 drawStyle setTexture 0
RadiossSurfaceView9 drawStyle setAlphaMode 1
RadiossSurfaceView9 drawStyle setNormalBinding 0
RadiossSurfaceView9 drawStyle setCullingMode 0
RadiossSurfaceView9 drawStyle setSortingMode 1
RadiossSurfaceView9 selectionMode setValue 0 0
RadiossSurfaceView9 Patch setMinMax 0 1
RadiossSurfaceView9 Patch setButtons 1
RadiossSurfaceView9 Patch setIncrement 1
RadiossSurfaceView9 Patch setValue 0
RadiossSurfaceView9 Patch setSubMinMax 0 1
RadiossSurfaceView9 BoundaryId setValue 0 -1
RadiossSurfaceView9 materials setValue 0 1
RadiossSurfaceView9 materials setValue 1 0
RadiossSurfaceView9 colorMode setValue 0
RadiossSurfaceView9 baseTrans setMinMax 0 1
RadiossSurfaceView9 baseTrans setButtons 0
RadiossSurfaceView9 baseTrans setIncrement 0.1
RadiossSurfaceView9 baseTrans setValue 0.8
RadiossSurfaceView9 baseTrans setSubMinMax 0 1
RadiossSurfaceView9 VRMode setValue 0 0
RadiossSurfaceView9 fire
RadiossSurfaceView9 hideBox 1
{RadiossSurfaceView9} selectTriangles zab HIJMPLPPHPBEIMICFBEAAPMAAAAEAAJPJOGIMC
RadiossSurfaceView9 fire
RadiossSurfaceView9 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView10}
RadiossSurfaceView10 setIconPosition 373 280
RadiossSurfaceView10 data connect LC16_opGlo_l_VT061079.surf
RadiossSurfaceView10 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView10 colormap setDefaultAlpha 0.500000
RadiossSurfaceView10 fire
RadiossSurfaceView10 drawStyle setValue 1
RadiossSurfaceView10 drawStyle setSpecularLighting 1
RadiossSurfaceView10 drawStyle setTexture 0
RadiossSurfaceView10 drawStyle setAlphaMode 1
RadiossSurfaceView10 drawStyle setNormalBinding 0
RadiossSurfaceView10 drawStyle setCullingMode 0
RadiossSurfaceView10 drawStyle setSortingMode 1
RadiossSurfaceView10 selectionMode setValue 0 0
RadiossSurfaceView10 Patch setMinMax 0 1
RadiossSurfaceView10 Patch setButtons 1
RadiossSurfaceView10 Patch setIncrement 1
RadiossSurfaceView10 Patch setValue 0
RadiossSurfaceView10 Patch setSubMinMax 0 1
RadiossSurfaceView10 BoundaryId setValue 0 -1
RadiossSurfaceView10 materials setValue 0 1
RadiossSurfaceView10 materials setValue 1 0
RadiossSurfaceView10 colorMode setValue 0
RadiossSurfaceView10 baseTrans setMinMax 0 1
RadiossSurfaceView10 baseTrans setButtons 0
RadiossSurfaceView10 baseTrans setIncrement 0.1
RadiossSurfaceView10 baseTrans setValue 0.8
RadiossSurfaceView10 baseTrans setSubMinMax 0 1
RadiossSurfaceView10 VRMode setValue 0 0
RadiossSurfaceView10 fire
RadiossSurfaceView10 hideBox 1
{RadiossSurfaceView10} selectTriangles zab HIJMONMBABABAAAAAIMDKAPHCPDNBDNIAACIAAAADONLDKFGCACEFL
RadiossSurfaceView10 fire
RadiossSurfaceView10 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView11}
RadiossSurfaceView11 setIconPosition 373 310
RadiossSurfaceView11 data connect all_glomeruli_R.surf
RadiossSurfaceView11 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView11 colormap setDefaultAlpha 0.500000
RadiossSurfaceView11 fire
RadiossSurfaceView11 drawStyle setValue 1
RadiossSurfaceView11 drawStyle setSpecularLighting 1
RadiossSurfaceView11 drawStyle setTexture 0
RadiossSurfaceView11 drawStyle setAlphaMode 1
RadiossSurfaceView11 drawStyle setNormalBinding 0
RadiossSurfaceView11 drawStyle setCullingMode 0
RadiossSurfaceView11 drawStyle setSortingMode 1
RadiossSurfaceView11 selectionMode setValue 0 0
RadiossSurfaceView11 Patch setMinMax 0 1
RadiossSurfaceView11 Patch setButtons 1
RadiossSurfaceView11 Patch setIncrement 1
RadiossSurfaceView11 Patch setValue 0
RadiossSurfaceView11 Patch setSubMinMax 0 1
RadiossSurfaceView11 BoundaryId setValue 0 -1
RadiossSurfaceView11 materials setValue 0 1
RadiossSurfaceView11 materials setValue 1 0
RadiossSurfaceView11 colorMode setValue 0
RadiossSurfaceView11 baseTrans setMinMax 0 1
RadiossSurfaceView11 baseTrans setButtons 0
RadiossSurfaceView11 baseTrans setIncrement 0.1
RadiossSurfaceView11 baseTrans setValue 0.8
RadiossSurfaceView11 baseTrans setSubMinMax 0 1
RadiossSurfaceView11 VRMode setValue 0 0
RadiossSurfaceView11 fire
RadiossSurfaceView11 hideBox 1
{RadiossSurfaceView11} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBCIBILJIAABAIAAGGIIABGC
RadiossSurfaceView11 fire
RadiossSurfaceView11 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView12}
RadiossSurfaceView12 setIconPosition 373 340
RadiossSurfaceView12 data connect LC18_opGlo_l_GMR92B11.surf
RadiossSurfaceView12 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView12 colormap setDefaultAlpha 0.500000
RadiossSurfaceView12 fire
RadiossSurfaceView12 drawStyle setValue 1
RadiossSurfaceView12 drawStyle setSpecularLighting 1
RadiossSurfaceView12 drawStyle setTexture 0
RadiossSurfaceView12 drawStyle setAlphaMode 1
RadiossSurfaceView12 drawStyle setNormalBinding 0
RadiossSurfaceView12 drawStyle setCullingMode 0
RadiossSurfaceView12 drawStyle setSortingMode 1
RadiossSurfaceView12 selectionMode setValue 0 0
RadiossSurfaceView12 Patch setMinMax 0 1
RadiossSurfaceView12 Patch setButtons 1
RadiossSurfaceView12 Patch setIncrement 1
RadiossSurfaceView12 Patch setValue 0
RadiossSurfaceView12 Patch setSubMinMax 0 1
RadiossSurfaceView12 BoundaryId setValue 0 -1
RadiossSurfaceView12 materials setValue 0 1
RadiossSurfaceView12 materials setValue 1 0
RadiossSurfaceView12 colorMode setValue 0
RadiossSurfaceView12 baseTrans setMinMax 0 1
RadiossSurfaceView12 baseTrans setButtons 0
RadiossSurfaceView12 baseTrans setIncrement 0.1
RadiossSurfaceView12 baseTrans setValue 0.8
RadiossSurfaceView12 baseTrans setSubMinMax 0 1
RadiossSurfaceView12 VRMode setValue 0 0
RadiossSurfaceView12 fire
RadiossSurfaceView12 hideBox 1
{RadiossSurfaceView12} selectTriangles zab HIJMPLPPHPBEIMICFBDANAIAABIIABBPJMIHKD
RadiossSurfaceView12 fire
RadiossSurfaceView12 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView13}
RadiossSurfaceView13 setIconPosition 373 370
RadiossSurfaceView13 data connect LC20_opGlo_l_VT025718.surf
RadiossSurfaceView13 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView13 colormap setDefaultAlpha 0.500000
RadiossSurfaceView13 fire
RadiossSurfaceView13 drawStyle setValue 1
RadiossSurfaceView13 drawStyle setSpecularLighting 1
RadiossSurfaceView13 drawStyle setTexture 0
RadiossSurfaceView13 drawStyle setAlphaMode 1
RadiossSurfaceView13 drawStyle setNormalBinding 0
RadiossSurfaceView13 drawStyle setCullingMode 0
RadiossSurfaceView13 drawStyle setSortingMode 1
RadiossSurfaceView13 selectionMode setValue 0 0
RadiossSurfaceView13 Patch setMinMax 0 1
RadiossSurfaceView13 Patch setButtons 1
RadiossSurfaceView13 Patch setIncrement 1
RadiossSurfaceView13 Patch setValue 0
RadiossSurfaceView13 Patch setSubMinMax 0 1
RadiossSurfaceView13 BoundaryId setValue 0 -1
RadiossSurfaceView13 materials setValue 0 1
RadiossSurfaceView13 materials setValue 1 0
RadiossSurfaceView13 colorMode setValue 0
RadiossSurfaceView13 baseTrans setMinMax 0 1
RadiossSurfaceView13 baseTrans setButtons 0
RadiossSurfaceView13 baseTrans setIncrement 0.1
RadiossSurfaceView13 baseTrans setValue 0.8
RadiossSurfaceView13 baseTrans setSubMinMax 0 1
RadiossSurfaceView13 VRMode setValue 0 0
RadiossSurfaceView13 fire
RadiossSurfaceView13 hideBox 1
{RadiossSurfaceView13} selectTriangles zab HIJMPLPPHPOEAAAGAGPOPPAAIFKPOMCD
RadiossSurfaceView13 fire
RadiossSurfaceView13 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView14}
RadiossSurfaceView14 setIconPosition 373 400
RadiossSurfaceView14 data connect LC21_opGlo_l_GMR85F11.surf
RadiossSurfaceView14 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView14 colormap setDefaultAlpha 0.500000
RadiossSurfaceView14 fire
RadiossSurfaceView14 drawStyle setValue 1
RadiossSurfaceView14 drawStyle setSpecularLighting 1
RadiossSurfaceView14 drawStyle setTexture 0
RadiossSurfaceView14 drawStyle setAlphaMode 1
RadiossSurfaceView14 drawStyle setNormalBinding 0
RadiossSurfaceView14 drawStyle setCullingMode 0
RadiossSurfaceView14 drawStyle setSortingMode 1
RadiossSurfaceView14 selectionMode setValue 0 0
RadiossSurfaceView14 Patch setMinMax 0 1
RadiossSurfaceView14 Patch setButtons 1
RadiossSurfaceView14 Patch setIncrement 1
RadiossSurfaceView14 Patch setValue 0
RadiossSurfaceView14 Patch setSubMinMax 0 1
RadiossSurfaceView14 BoundaryId setValue 0 -1
RadiossSurfaceView14 materials setValue 0 1
RadiossSurfaceView14 materials setValue 1 0
RadiossSurfaceView14 colorMode setValue 0
RadiossSurfaceView14 baseTrans setMinMax 0 1
RadiossSurfaceView14 baseTrans setButtons 0
RadiossSurfaceView14 baseTrans setIncrement 0.1
RadiossSurfaceView14 baseTrans setValue 0.8
RadiossSurfaceView14 baseTrans setSubMinMax 0 1
RadiossSurfaceView14 VRMode setValue 0 0
RadiossSurfaceView14 fire
RadiossSurfaceView14 hideBox 1
{RadiossSurfaceView14} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKAGAPGAAAGCAAHKPLJCKG
RadiossSurfaceView14 fire
RadiossSurfaceView14 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView15}
RadiossSurfaceView15 setIconPosition 373 430
RadiossSurfaceView15 data connect LC22_LPLC4_opGlo_l_VT058688.surf
RadiossSurfaceView15 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView15 colormap setDefaultAlpha 0.500000
RadiossSurfaceView15 fire
RadiossSurfaceView15 drawStyle setValue 1
RadiossSurfaceView15 drawStyle setSpecularLighting 1
RadiossSurfaceView15 drawStyle setTexture 0
RadiossSurfaceView15 drawStyle setAlphaMode 1
RadiossSurfaceView15 drawStyle setNormalBinding 0
RadiossSurfaceView15 drawStyle setCullingMode 0
RadiossSurfaceView15 drawStyle setSortingMode 1
RadiossSurfaceView15 selectionMode setValue 0 0
RadiossSurfaceView15 Patch setMinMax 0 1
RadiossSurfaceView15 Patch setButtons 1
RadiossSurfaceView15 Patch setIncrement 1
RadiossSurfaceView15 Patch setValue 0
RadiossSurfaceView15 Patch setSubMinMax 0 1
RadiossSurfaceView15 BoundaryId setValue 0 -1
RadiossSurfaceView15 materials setValue 0 1
RadiossSurfaceView15 materials setValue 1 0
RadiossSurfaceView15 colorMode setValue 0
RadiossSurfaceView15 baseTrans setMinMax 0 1
RadiossSurfaceView15 baseTrans setButtons 0
RadiossSurfaceView15 baseTrans setIncrement 0.1
RadiossSurfaceView15 baseTrans setValue 0.8
RadiossSurfaceView15 baseTrans setSubMinMax 0 1
RadiossSurfaceView15 VRMode setValue 0 0
RadiossSurfaceView15 fire
RadiossSurfaceView15 hideBox 1
{RadiossSurfaceView15} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBCICAAPDADADAPMAHAAKLADDPBG
RadiossSurfaceView15 fire
RadiossSurfaceView15 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView16}
RadiossSurfaceView16 setIconPosition 373 460
RadiossSurfaceView16 data connect LC24_opGlo_l_VT038216.surf
RadiossSurfaceView16 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView16 colormap setDefaultAlpha 0.500000
RadiossSurfaceView16 fire
RadiossSurfaceView16 drawStyle setValue 1
RadiossSurfaceView16 drawStyle setSpecularLighting 1
RadiossSurfaceView16 drawStyle setTexture 0
RadiossSurfaceView16 drawStyle setAlphaMode 1
RadiossSurfaceView16 drawStyle setNormalBinding 0
RadiossSurfaceView16 drawStyle setCullingMode 0
RadiossSurfaceView16 drawStyle setSortingMode 1
RadiossSurfaceView16 selectionMode setValue 0 0
RadiossSurfaceView16 Patch setMinMax 0 1
RadiossSurfaceView16 Patch setButtons 1
RadiossSurfaceView16 Patch setIncrement 1
RadiossSurfaceView16 Patch setValue 0
RadiossSurfaceView16 Patch setSubMinMax 0 1
RadiossSurfaceView16 BoundaryId setValue 0 -1
RadiossSurfaceView16 materials setValue 0 1
RadiossSurfaceView16 materials setValue 1 0
RadiossSurfaceView16 colorMode setValue 0
RadiossSurfaceView16 baseTrans setMinMax 0 1
RadiossSurfaceView16 baseTrans setButtons 0
RadiossSurfaceView16 baseTrans setIncrement 0.1
RadiossSurfaceView16 baseTrans setValue 0.8
RadiossSurfaceView16 baseTrans setSubMinMax 0 1
RadiossSurfaceView16 VRMode setValue 0 0
RadiossSurfaceView16 fire
RadiossSurfaceView16 hideBox 1
{RadiossSurfaceView16} selectTriangles zab HIJMPLPPHPBEIMICFBDABAIAIBIBIBBPAAPDGOHIMB
RadiossSurfaceView16 fire
RadiossSurfaceView16 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView17}
RadiossSurfaceView17 setIconPosition 373 490
RadiossSurfaceView17 data connect LPC1_opGlo_l_GMR77A06.surf
RadiossSurfaceView17 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView17 colormap setDefaultAlpha 0.500000
RadiossSurfaceView17 fire
RadiossSurfaceView17 drawStyle setValue 1
RadiossSurfaceView17 drawStyle setSpecularLighting 1
RadiossSurfaceView17 drawStyle setTexture 0
RadiossSurfaceView17 drawStyle setAlphaMode 1
RadiossSurfaceView17 drawStyle setNormalBinding 0
RadiossSurfaceView17 drawStyle setCullingMode 0
RadiossSurfaceView17 drawStyle setSortingMode 1
RadiossSurfaceView17 selectionMode setValue 0 0
RadiossSurfaceView17 Patch setMinMax 0 1
RadiossSurfaceView17 Patch setButtons 1
RadiossSurfaceView17 Patch setIncrement 1
RadiossSurfaceView17 Patch setValue 0
RadiossSurfaceView17 Patch setSubMinMax 0 1
RadiossSurfaceView17 BoundaryId setValue 0 -1
RadiossSurfaceView17 materials setValue 0 1
RadiossSurfaceView17 materials setValue 1 0
RadiossSurfaceView17 colorMode setValue 0
RadiossSurfaceView17 baseTrans setMinMax 0 1
RadiossSurfaceView17 baseTrans setButtons 0
RadiossSurfaceView17 baseTrans setIncrement 0.1
RadiossSurfaceView17 baseTrans setValue 0.8
RadiossSurfaceView17 baseTrans setSubMinMax 0 1
RadiossSurfaceView17 VRMode setValue 0 0
RadiossSurfaceView17 fire
RadiossSurfaceView17 hideBox 1
{RadiossSurfaceView17} selectTriangles zab HIJMPLPPHPBEIMICFBDAJCABADADADDPAAHFIDAAEI
RadiossSurfaceView17 fire
RadiossSurfaceView17 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView18}
RadiossSurfaceView18 setIconPosition 373 520
RadiossSurfaceView18 data connect LPLC1_opGlo_l_GMR36B06.surf
RadiossSurfaceView18 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView18 colormap setDefaultAlpha 0.500000
RadiossSurfaceView18 fire
RadiossSurfaceView18 drawStyle setValue 1
RadiossSurfaceView18 drawStyle setSpecularLighting 1
RadiossSurfaceView18 drawStyle setTexture 0
RadiossSurfaceView18 drawStyle setAlphaMode 1
RadiossSurfaceView18 drawStyle setNormalBinding 0
RadiossSurfaceView18 drawStyle setCullingMode 0
RadiossSurfaceView18 drawStyle setSortingMode 1
RadiossSurfaceView18 selectionMode setValue 0 0
RadiossSurfaceView18 Patch setMinMax 0 1
RadiossSurfaceView18 Patch setButtons 1
RadiossSurfaceView18 Patch setIncrement 1
RadiossSurfaceView18 Patch setValue 0
RadiossSurfaceView18 Patch setSubMinMax 0 1
RadiossSurfaceView18 BoundaryId setValue 0 -1
RadiossSurfaceView18 materials setValue 0 1
RadiossSurfaceView18 materials setValue 1 0
RadiossSurfaceView18 colorMode setValue 0
RadiossSurfaceView18 baseTrans setMinMax 0 1
RadiossSurfaceView18 baseTrans setButtons 0
RadiossSurfaceView18 baseTrans setIncrement 0.1
RadiossSurfaceView18 baseTrans setValue 0.8
RadiossSurfaceView18 baseTrans setSubMinMax 0 1
RadiossSurfaceView18 VRMode setValue 0 0
RadiossSurfaceView18 fire
RadiossSurfaceView18 hideBox 1
{RadiossSurfaceView18} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKEGMBCIBIHMIABPIIABBGLPJMMH
RadiossSurfaceView18 fire
RadiossSurfaceView18 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView19}
RadiossSurfaceView19 setIconPosition 373 550
RadiossSurfaceView19 data connect LPLC2_opGlo_l_VT007194.surf
RadiossSurfaceView19 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView19 colormap setDefaultAlpha 0.500000
RadiossSurfaceView19 fire
RadiossSurfaceView19 drawStyle setValue 1
RadiossSurfaceView19 drawStyle setSpecularLighting 1
RadiossSurfaceView19 drawStyle setTexture 0
RadiossSurfaceView19 drawStyle setAlphaMode 1
RadiossSurfaceView19 drawStyle setNormalBinding 0
RadiossSurfaceView19 drawStyle setCullingMode 0
RadiossSurfaceView19 drawStyle setSortingMode 1
RadiossSurfaceView19 selectionMode setValue 0 0
RadiossSurfaceView19 Patch setMinMax 0 1
RadiossSurfaceView19 Patch setButtons 1
RadiossSurfaceView19 Patch setIncrement 1
RadiossSurfaceView19 Patch setValue 0
RadiossSurfaceView19 Patch setSubMinMax 0 1
RadiossSurfaceView19 BoundaryId setValue 0 -1
RadiossSurfaceView19 materials setValue 0 1
RadiossSurfaceView19 materials setValue 1 0
RadiossSurfaceView19 colorMode setValue 0
RadiossSurfaceView19 baseTrans setMinMax 0 1
RadiossSurfaceView19 baseTrans setButtons 0
RadiossSurfaceView19 baseTrans setIncrement 0.1
RadiossSurfaceView19 baseTrans setValue 0.8
RadiossSurfaceView19 baseTrans setSubMinMax 0 1
RadiossSurfaceView19 VRMode setValue 0 0
RadiossSurfaceView19 fire
RadiossSurfaceView19 hideBox 1
{RadiossSurfaceView19} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKAIADAGCAAAAAEOMECIBB
RadiossSurfaceView19 fire
RadiossSurfaceView19 setViewerMask 65535

set hideNewModules 0
create HxDisplaySurface {RadiossSurfaceView20}
RadiossSurfaceView20 setIconPosition 373 580
RadiossSurfaceView20 data connect LPLC3_opGlo_l_VT044492.surf
RadiossSurfaceView20 colormap setDefaultColor 1 0.1 0.1
RadiossSurfaceView20 colormap setDefaultAlpha 0.500000
RadiossSurfaceView20 fire
RadiossSurfaceView20 drawStyle setValue 1
RadiossSurfaceView20 drawStyle setSpecularLighting 1
RadiossSurfaceView20 drawStyle setTexture 0
RadiossSurfaceView20 drawStyle setAlphaMode 1
RadiossSurfaceView20 drawStyle setNormalBinding 0
RadiossSurfaceView20 drawStyle setCullingMode 0
RadiossSurfaceView20 drawStyle setSortingMode 1
RadiossSurfaceView20 selectionMode setValue 0 0
RadiossSurfaceView20 Patch setMinMax 0 1
RadiossSurfaceView20 Patch setButtons 1
RadiossSurfaceView20 Patch setIncrement 1
RadiossSurfaceView20 Patch setValue 0
RadiossSurfaceView20 Patch setSubMinMax 0 1
RadiossSurfaceView20 BoundaryId setValue 0 -1
RadiossSurfaceView20 materials setValue 0 1
RadiossSurfaceView20 materials setValue 1 0
RadiossSurfaceView20 colorMode setValue 0
RadiossSurfaceView20 baseTrans setMinMax 0 1
RadiossSurfaceView20 baseTrans setButtons 0
RadiossSurfaceView20 baseTrans setIncrement 0.1
RadiossSurfaceView20 baseTrans setValue 0.8
RadiossSurfaceView20 baseTrans setSubMinMax 0 1
RadiossSurfaceView20 VRMode setValue 0 0
RadiossSurfaceView20 fire
RadiossSurfaceView20 hideBox 1
{RadiossSurfaceView20} selectTriangles zab HIJMPLPPHPBEIMICFBDAAKOIALBIBIPOPPAHAAMJJHHFMD
RadiossSurfaceView20 fire
RadiossSurfaceView20 setViewerMask 65535

set hideNewModules 0


viewer 0 setCameraPosition 233.566 147.614 -442.036
viewer 0 setCameraOrientation 0.995529 0.0415665 -0.0848168 3.13338
viewer 0 setCameraFocalDistance 518.733
viewer 0 setAutoRedraw 1
viewer 0 redraw
