# Generates a short synthetic teacher/student classroom dialogue using the
# built-in Windows SAPI voices (Zira = Teacher, David = Student). This is
# the audio bundled with the repo and used by the public live demo, so the
# real classroom recordings supplied for this assignment (which include
# children's voices, GPS-tagged school locations, and a teacher's real
# name) never need to be published anywhere.
Add-Type -AssemblyName System.Speech

$outDir = Join-Path $PSScriptRoot "..\data\sample"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$segDir = Join-Path $outDir "_segments"
New-Item -ItemType Directory -Force -Path $segDir | Out-Null

# (speaker, voice, rate, text, pauseAfterMs)
$lines = @(
    @("Teacher","Zira",0,"Good morning everyone. Today we are going to learn about photosynthesis.",600),
    @("Teacher","Zira",0,"Can anyone tell me, what do plants need to make their own food?",1200),
    @("Student","David",0,"Sunlight, water, and carbon dioxide.",400),
    @("Teacher","Zira",0,"Excellent answer! And where does photosynthesis take place in the plant cell?",1000),
    @("Student","David",0,"In the chloroplast.",400),
    @("Teacher","Zira",0,"Correct. Now, why do you think leaves are green?",2500),
    @("Teacher","Zira",0,"It's because of a pigment called chlorophyll. Chlorophyll absorbs red and blue light and reflects green light.",600),
    @("Teacher","Zira",0,"Does everyone understand this so far?",800),
    @("Student","David",0,"Yes ma'am.",400),
    @("Teacher","Zira",0,"Great. Let's move to the next topic. What is the difference between photosynthesis and respiration?",1500),
    @("Student","David",0,"Respiration uses oxygen and releases carbon dioxide, but photosynthesis uses carbon dioxide and releases oxygen.",400),
    @("Teacher","Zira",0,"Very good, that is a complete answer.",600),
    @("Teacher","Zira",0,"Now let's do a quick activity. Open your textbooks to page forty two.",3500),
    @("Teacher","Zira",0,"Has everyone found the diagram?",800),
    @("Student","David",0,"Yes.",400),
    @("Teacher","Zira",0,"Good. Please label the parts of the leaf shown in the diagram.",400)
)

$i = 0
$fileList = @()
foreach ($line in $lines) {
    $speaker, $voiceName, $rate, $text, $pauseMs = $line
    $i++
    $wavPath = Join-Path $segDir ("{0:d2}_{1}.wav" -f $i, $speaker)

    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $voiceFull = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Name -like "*$voiceName*" } | Select-Object -First 1
    $synth.SelectVoice($voiceFull.VoiceInfo.Name)
    $synth.Rate = $rate
    $synth.SetOutputToWaveFile($wavPath)
    $synth.Speak($text)
    $synth.Dispose()
    $fileList += $wavPath

    if ($pauseMs -gt 0) {
        $silencePath = Join-Path $segDir ("{0:d2}_silence.wav" -f $i)
        & ffmpeg -y -f lavfi -i "anullsrc=r=16000:cl=mono" -t ([double]$pauseMs/1000.0) -q:a 9 "$silencePath" *> $null
        $fileList += $silencePath
    }
}

$concatListPath = Join-Path $segDir "concat_list.txt"
$concatLines = $fileList | ForEach-Object { "file '" + ($_ -replace '\\','/') + "'" }
Set-Content -Path $concatListPath -Value $concatLines -Encoding ascii

$finalWav = Join-Path $outDir "classroom_sample.wav"
$finalMp3 = Join-Path $outDir "classroom_sample.mp3"
& ffmpeg -y -f concat -safe 0 -i "$concatListPath" -ar 16000 -ac 1 "$finalWav" *> $null
if (-not (Test-Path $finalWav)) { Write-Error "concat step failed, see $concatListPath"; exit 1 }
& ffmpeg -y -i "$finalWav" -codec:a libmp3lame -qscale:a 4 "$finalMp3" *> $null
if (-not (Test-Path $finalMp3)) { Write-Error "mp3 encode step failed"; exit 1 }

Remove-Item -Recurse -Force $segDir
Remove-Item -Force $finalWav

Write-Host "Generated sample audio: $finalMp3"
