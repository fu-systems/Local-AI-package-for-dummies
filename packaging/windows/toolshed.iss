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
// Remove Toolshed completely, keeping only the models.
//
// The data root lives outside {app} by design, and the uninstaller used to
// leave every byte of it -- so a reinstall landed on top of a half-finished
// one and inherited its problems: a Python workspace built by an interrupted
// run, a manifest describing files that were gone, an engine tree from a
// different version.
//
// Models are the exception, and only models. They are tens of gigabytes and
// re-downloading them is the slowest part of any reinstall.
//
// Kept in step with packaging/linux/uninstall.sh by a test; the two must name
// the same folder to keep and find the data root the same way.

function DataRoot(): String;
begin
  Result := GetEnv('TOOLSHED_ROOT');
  if Result = '' then
    Result := ExpandConstant('{%SYSTEMDRIVE|C:}\Toolshed');
end;

procedure PurgeDataRoot();
var
  Root, Keep, Item: String;
  Search: TFindRec;
begin
  Root := DataRoot();
  Keep := 'models';

  // Deleting trees: refuse anywhere that is not plainly our own folder. A
  // missing SYSTEMDRIVE would otherwise leave a root-relative path here.
  if (Root = '') or (Pos('Toolshed', Root) = 0) then
    Exit;
  if not DirExists(Root) then
    Exit;

  if not DirExists(Root + '\' + Keep) then
  begin
    DelTree(Root, True, True, True);
    Exit;
  end;

  if FindFirst(Root + '\*', Search) then
  begin
    try
      repeat
        if (Search.Name = '.') or (Search.Name = '..') then
          Continue;
        if CompareText(Search.Name, Keep) = 0 then
          Continue;
        Item := Root + '\' + Search.Name;
        // DirExists rather than the attribute bits: it is documented, and
        // this file cannot be compiled or run anywhere but Windows, so it
        // uses only functions whose behaviour is not in question.
        if DirExists(Item) then
          DelTree(Item, True, True, True)
        else
          DeleteFile(Item);
      until not FindNext(Search);
    finally
      FindClose(Search);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurStep: TUninstallStep);
var
  Root: String;
begin
  if CurStep = usPostUninstall then
  begin
    Root := DataRoot();
    PurgeDataRoot();
    if DirExists(Root + '\models') then
      MsgBox('Toolshed has been removed.'#13#10#13#10 +
             'Your models were kept, in ' + Root + '\models.'#13#10 +
             'Nothing else was left anywhere. Delete that folder too if you ' +
             'want the disk space back.',
             mbInformation, MB_OK)
    else
      MsgBox('Toolshed has been removed.'#13#10#13#10 +
             'Nothing was left anywhere.',
             mbInformation, MB_OK);
  end;
end;
