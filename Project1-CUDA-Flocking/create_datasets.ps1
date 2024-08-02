cmake --build build -j --config Release

$startExponent = 10
$endExponent = 20

$programPath = ".\install\bin\Release\cis565_boids.exe"

for ($i = $startExponent; $i -le $endExponent; $i++) {
   
    $N = [math]::Pow(2, $i)
    
    $arguments = @("-n", "$N", "-v", "-s")

    & $programPath $arguments

}
