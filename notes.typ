#set page(height: auto)
// Number equations
#set math.equation(numbering: "(1)")

#let lv = $l_"v"$
#let Lv = $L_"v"$
#let Lvz = $L_"vz"$
#let Evd = $E_"vd"$

*The Perez All-Weather model.*

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

#pagebreak()

*Preetham model.*

_Restatement of the Perez All-Weather model._

$
  cal(F)(theta, gamma) = (1 + A e^(B slash cos theta)) (1 + C e^(D gamma) + E cos^2 gamma).
$
$
  Y = Y_z cal(F)(theta, gamma) slash cal(F)(0, theta_s).
$

_Chromatic model extension._

$
  x = x_z (cal(F)(theta, gamma)) / (cal(F)(0, theta_s)), quad "and" quad
  y = y_z (cal(F)(theta, gamma)) / (cal(F)(0, theta_s)).
$

_Perez coeffiecients from turbidity and solar zenith angle._

$
  mat(delim: "[", A_Y; B_Y; C_Y; D_Y; E_Y) = mat(delim: "[",
    &0.1787, - &1.4630;
   -&0.3554, &0.4275;
   -&0.0227, &5.3251;
    &0.1206, - &2.5771;
   -&0.0670, &0.3703) mat(delim: "[", T; 1).
$
$
  mat(delim: "[", A_x; B_x; C_x; D_x; E_x) = mat(delim: "[",
   -&0.0193, - &0.2592;
   -&0.0665, &0.0008;
   -&0.0004, &0.2125;
   -&0.0641, - &0.8989;
   -&0.0033, &0.0452) mat(delim: "[", T; 1).
$
$
  mat(delim: "[", A_y; B_y; C_y; D_y; E_y) = mat(delim: "[",
   -&0.0167, - &0.2608;
   -&0.0950, &0.0092;
   -&0.0079, &0.2102;
   -&0.0441, - &1.6537;
   -&0.0109, &0.0529) mat(delim: "[", T; 1).
$

_Absolute zenith luminance and chromaticity._

$
  Y_z = (4.0453 T - 4.9710) tan chi - 0.2155 T + 2.4192.
$
$
  chi = ( 4 / 9 - T / 120 ) ( pi - 2 theta_s ).
$
$
  x_z = mat(delim: "[",
    T^2, T, 1) mat(delim: "[",
    &0.0017, - &0.0037, &0.0021, &0.000;
    -&0.0290, &0.0638, - &0.0320, &0.0039;
    &0.1169, - &0.2120, &0.0605, &0.2589) mat(delim: "[",
    theta_s^3; theta_s^2; theta_s; 1).
$
$
  y_z = mat(delim: "[",
    T^2, T, 1) mat(delim: "[",
    &0.0028, - &0.0061, &0.0032, &0.000;
   -&0.0421, &0.0897, - &0.0415, &0.0052;
    &0.1535, - &0.2676, &0.0667, &0.2669) mat(delim: "[",
    theta_s^3; theta_s^2; theta_s; 1).
$

#pagebreak()

*Additional computations.*

_Discretization in equirectangular coordinates._

$
  (i, j) in {0, ..., W - 1} times {0, ..., floor(H slash 2) - 1} -> (u, v) in [0, 1) times (0, 1), \
  u = (i + 0.5) / W, quad "and" quad v = (2j + 1) / H.
$

_Equirectangular projection (From Wikipedia)._

$
  (u, v) in [0, 1) times (0, 1) <-> (psi, phi.alt) in [-pi, pi) times (0, pi slash 2), \
  psi = 2 pi u, quad "and" quad phi.alt = pi / 2 ( 1 - v ).
$

_Conversion between standard and non-standard spherical coordinates._

$
  (psi, phi.alt, psi_s, phi.alt_s) in [-pi, pi) times (0, pi slash 2) times [-pi, pi) times (0, pi slash 2) \
  -> \
  (theta, theta_s, gamma) in (0, pi slash 2) times (0, pi slash 2) times [0, pi).
$

_Zenith angles._
$
  theta = pi / 2 - phi.alt, quad "and" quad theta_s = pi / 2 - phi.alt_s.
$

_Spherical distance in spherical coordinates (from Wikipedia)._

$
  gamma = arccos( sin phi.alt sin phi.alt_s + cos phi.alt cos phi.alt_s cos( Delta psi ) ).
$

_Haversine formula (from Wikipedia)._

$
  gamma = "archav"("hav"(Delta phi.alt) + (1 - "hav"(phi.alt + phi.alt_s)) "hav"(Delta psi) ).
$

_Vincenty formula (from Wikipedia)._

$
  gamma = "atan2"( sqrt( (cos phi.alt_s sin( Delta psi ))^2 + ( cos phi.alt sin phi.alt_s - sin phi.alt cos phi.alt_s cos( Delta psi ) )^2 ), \
    sin phi.alt sin phi.alt_s + cos phi.alt cos phi.alt_s cos( Delta psi ) ).
$

_Conversion from CIE xyY to CIE XYZ (from Wikipedia)._

$
  (x, y) in [0, 1] times (0, 1] <-> (X, Z) in [0, infinity) times [0, infinity), \
  X = Y / y x, quad "and" quad Z = Y / y (1 - x - y).
$

_Conversion from CIE XYZ to linear sRGB (from Wikipedia)._

$
  mat(delim: "[", R; G; B) = mat(delim: "[",
  +&3.2406255, -&1.5372073, -&0.4986286;
  -&0.9689307, +&1.8757561, +&0.0415175;
  +&0.0557101, -&0.2040211, +&1.0569959
) mat(delim: "[",
  X; Y; Z)
$

_Conversion from linear sRGB to sRGB (from Wikipedia)._

$
  R' = cases(
    12.92 R quad &"if" R <= 0.0031308",",
    1.055 R^(1 slash 2.4) - 0.055 quad &"otherwise".
  ), quad #[and similarly for $G'$ and $B'$.]
$
