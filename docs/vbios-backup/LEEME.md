# Backups de vBIOS de las dos 3090

| archivo | qué es |
|---|---|
| `zotac_flash_completo.rom` | flash COMPLETO de la Zotac ANTES del flasheo (999.424 B, `94.02.42.80.9F`, fecha **01/06/21**), leído con `nvflash --save` |
| `pre_flash_0428.rom` | idéntico al anterior, tomado minutos antes de escribir |
| `zotac_trinity_oc_rebar_210305.rom` | el que se FLASHEÓ el 2026-08-17: mismo `94.02.42.80.9F` pero fecha **03/05/21**, con ReBAR. Extraído del instalador oficial `ZT-A30900J-10P_rebar.exe` (offset 992 de la sección `.data`) |
| `vbios_0000_0f_00.0.rom` / `vbios_0000_11_00.0.rom` | ventanas PCI-visibles (156.672 B) leídas por sysfs. **NO sirven para flashear**: son el 15% del firmware |

## Restaurar la Zotac

```bash
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo /home/usuario/Proyectos/temp/x64/nvflash --index=0 -6 pre_flash_0428.rom
```

`nvflash` necesita una terminal REAL: lee del tty y ninguna redirección de stdin
le sirve (`script`, pipes, heredocs — todo falla con "Reading from the keyboard
failed" o queda bloqueado en `read()`). Hay que correrlo a mano.

`--index=0` es la Zotac (`10DE:2204:19DA:1613`), `--index=1` la Gigabyte
(`1458:4043`). `nvflash` compara los IDs y se niega a escribir si no coinciden.

## Ojo

- Descargar/recargar los módulos deja una placa con `WPR2 already up` y
  `nvidia-smi -L` muestra una sola. Se arregla con
  `echo 1 > /sys/bus/pci/devices/0000:0f:00.0/reset`, sin reiniciar.
- `spark-dashboard` retiene `/dev/nvidia-uvm` y hay que pararlo antes.
- `nvidia-persistenced` tiene que estar arriba ANTES que los contenedores.
- ⚠️ Los ROMs de TechPowerUp son volcados de placas de usuarios: el del
  `210305` traía 370 bytes de datos ajenos en `0x4000`, una región que esta
  placa tiene sin programar. Por eso se usó el del instalador oficial.

Ver `../P2P-SIN-PARCHAR-EL-DRIVER.md`
