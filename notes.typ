#set page(height: auto)
// Number equations
#set math.equation(numbering: "(1)")

#let lv = $l_"v"$
#let Lv = $L_"v"$
#let Lvz = $L_"vz"$
#let Evd = $E_"vd"$

*Perez All-Weather.*

_Relative luminance._

$
lv = f(xi, gamma) = [1 + a exp(b slash cos xi)] dot [1 + c exp(d gamma) + e cos^2 gamma].
$

_Assumption._

$
b < 0.
$

_Luminance at the zenith._

$
lv("0°", gamma) = [ 1 + a exp(b) ] dot [1 + c exp(d gamma) + e cos^2 gamma].
$

_Luminance at the horizon._

$
f("90°", gamma) = 1 + c exp(d gamma) + e cos^2 gamma.
$

_Luminance of the sun._

$
f(xi, "0°") = [ 1 + a exp(b / cos xi) ] dot [ 1 + c + e ]
$

_Luminance of the sun at the zenith._

$
f("0°", "0°") = [ 1 + a exp(b) ] dot [ 1 + c + e ].
$

_Luminance of the sun at the horizon._

$
f("90°", "0°") = 1 + c + e.
$

_Absolute luminance from absolute luminance at zenith._

$
Lv = Lvz f(xi, gamma) slash f("0°", gamma).
$

_Absolute luminance from illuminance._

$
Lv = lv Evd ( integral_"sky" [ "lv"(xi, gamma) cos xi ] "d" omega)^(-1).
$
