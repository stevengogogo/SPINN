mkdir -p out
python advection2d.py |> out/advection2d.txt
python heat2d.py |> out/heat2d.txt
python poisson2d_sine.py |> out/poisson2d_sine.txt
python helmotz2d.py |> out/helmotz2d.txt