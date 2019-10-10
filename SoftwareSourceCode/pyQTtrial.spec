# -*- mode: python -*-

a = Analysis(['kevin_retinopathy_software.py'],
             pathex=['C:\\Users\\hk609\\Desktop\\Thesis_Kevin_Retino_EXE_Final'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)

ui_file = [('mainWindow.ui',
'C:\\Users\\hk609\\Desktop\\Thesis_Kevin_Retino_EXE_Final\\mainWindow.ui', 'DATA')]

ui_file2 = [('graphPage.ui',
'C:\\Users\\hk609\\Desktop\\Thesis_Kevin_Retino_EXE_Final\\graphPage.ui', 'DATA')]


pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='kevin_retinopathy_software.exe',
          debug=False,
          strip=None,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas + ui_file + ui_file2,
               strip=None,
               upx=True,
               name='kevin_retinopathy_software')
