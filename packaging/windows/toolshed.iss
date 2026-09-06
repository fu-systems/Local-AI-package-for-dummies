; Inno Setup script for Toolshed.
;
; Compiled by the build workflow with:
;   ISCC /DMyAppVersion=x.y.z /DMySourceDir=...\dist\toolshed /DMyRepoRoot=... /O out toolshed.iss

#define MyAppName      "Toolshed"
#define MyAppPublisher "fu.systems"
#define MyAppURL       "https://github.com/fu-systems/Local-AI-package-for-dummies"
#define MyAppExeName   "toolshed.exe"

[Setup]
; The identity of the installation. Generated once; NEVER change it. Changing
; it makes every future installer a separate product rather than an upgrade,
; and leaves the old entry in Add/Remove Programs forever.
AppId={{7E3A9C21-5B84-4F6D-9A17-2C8E40D5B913}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases

; The whole point: no UAC prompt, no administrator, so someone on a locked-down
; or family machine can install it. With `lowest`, {autopf} resolves to
; %LOCALAPPDATA%\Programs and Inno writes the uninstall entry to HKCU by
; itself -- do not hand-write a [Registry] entry for it.
PrivilegesRequired=lowest
; Empty: forbid /ALLUSERS. A per-machine install would land in a state the
; uninstaller and the data-root logic were never designed for.
PrivilegesRequiredOverridesAllowed=
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
AllowNoIcons=yes

; "x64" is deprecated since Inno Setup 6.3. x64compatible also matches Arm64
; Windows, where this x64 build runs under emulation -- slowly, but it runs,
; which beats refusing to install.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763

OutputBaseFilename=Toolshed-{#MyAppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#MyRepoRoot}\packaging\windows\toolshed.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
; CloseApplications=no on purpose. Inno's default (yes) runs a Restart Manager
; scan for processes holding files it is about to write. On a first install
; there is nothing to close, and that scan is a known way for a silent install
; to stall on an unattended machine -- which is exactly what happened in CI.
; Revisit only if in-place upgrades over a running Toolshed become a problem,
; and pair it with a bounded wait if so.
CloseApplications=no
RestartApplications=no
; No LicenseFile page on purpose: a wall of Apache-2.0 text is not what a
; nervous beginner needs in a wizard. The licence ships as a file in {app}.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#MyRepoRoot}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#MyRepoRoot}\NOTICE";  DestDir: "{app}"; DestName: "NOTICE.txt";  Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}";  Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; \
  Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: dirifempty;     Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurStep: TUninstallStep);
begin
  // The data root -- models and outputs, tens of gigabytes -- lives OUTSIDE
  // {app} by design and is never touched here. Say so out loud: someone who
  // uninstalls to reclaim space and finds 45 GB still on disk will reasonably
  // conclude the uninstaller is broken.
  if CurStep = usPostUninstall then
    MsgBox('Toolshed has been removed.'#13#10#13#10 +
           'Your models and the things you made were NOT deleted. They are still ' +
           'in the folder you chose during setup. Delete that folder yourself if ' +
           'you want the disk space back.',
           mbInformation, MB_OK);
end;
