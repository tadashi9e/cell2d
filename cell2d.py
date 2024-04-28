# -*- coding: utf-8; mode:python -*-
import argparse
import datetime
import math
import random
import cupy as xp
from cupyx.scipy import signal
import cv2
from PIL import Image
from typing import Any, Optional

REG_N = 0  # North
REG_E = 1  # East
REG_S = 2  # South
REG_W = 3  # West
REG_C = 4  # Center

class Field(object):
    def __init__(self, width: int, height: int, rule: int):
        self._width = width
        self._height = height
        self._rule = rule
        # Cells
        self._cells = xp.zeros((height, width), dtype = xp.bool_)
        self._prev_cells = xp.zeros((height, width), dtype = xp.bool_)
        # Veon Neumann neighborhood
        self._n = xp.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]], dtype = xp.uint64)
        self._e = xp.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]], dtype = xp.uint64)
        self._w = xp.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]], dtype = xp.uint64)
        self._s = xp.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]], dtype = xp.uint64)
    def init_random(self) -> None:
        self._cells = xp.select(
            [xp.random.randint(
                1 + 1, size = (self._height, self._width)) != 0],
            [xp.asarray(xp.bool_(True))],
            default = xp.bool_(False))
    def mask(self) -> None:
        m = xp.zeros((self._height, self._width), dtype = xp.bool_)
        m[(self._height // 3):(self._height * 2 // 3),
          (self._width // 3):(self._width * 2 // 3)] = True
        self._cells = xp.logical_and(self._cells, m)
                      
    def get_current_bgr_image(self, cell_size: int) -> Any:
        # Prev Current  BG
        # -----------------
        # F    F          0                       -> black
        # F    T          0                       -> red
        # T    F          0                       -> red
        # T    T        255 (cell is stable true) -> white
        bg = xp.select(
            [xp.logical_and(self._cells, self._prev_cells)],
            [xp.asarray(xp.uint8(255))],
            default = xp.uint8(0))
        # Prev Current   R
        # -----------------
        # F    F          0                       -> black
        # F    T        255 (cell changed)        -> red
        # T    F        255 (cell changed)        -> red
        # T    T        255 (cell is stable true) -> white
        r = xp.select(
            [xp.logical_or(self._cells, self._prev_cells)],
            [xp.asarray(xp.uint8(255))],
            default = xp.uint8(0))
        if cell_size != 1:
            bg = bg.repeat(cell_size, axis = 0).repeat(cell_size, axis = 1)
            r = r.repeat(cell_size, axis = 0).repeat(cell_size, axis = 1)
        img = xp.zeros([self._height * cell_size, self._width * cell_size, 3],
                       dtype = xp.uint8)
        img[:,:,0] = bg
        img[:,:,1] = bg
        img[:,:,2] = r
        return xp.asnumpy(img)
    def update_cells(self) -> None:
        self._prev_cells = self._cells
        icells = xp.select([self._cells], [xp.asarray(1)], default = 0)
        r = (
            # N
            xp.select(
                [signal.convolve2d(icells, self._n,
                                   mode = 'same', boundary = 'wrap') != 0],
                [xp.asanyarray(1 << REG_N)],
                default = 0) +
            # E
            xp.select(
                [signal.convolve2d(icells, self._e,
                                   mode = 'same', boundary = 'wrap') != 0],
                [xp.asanyarray(1 << REG_E)],
                default = 0) +
            # W
            xp.select(
                [signal.convolve2d(icells, self._w,
                                   mode = 'same', boundary = 'wrap') != 0],
                [xp.asanyarray(1 << REG_W)],
                default = 0) +
            # S
            xp.select(
                [signal.convolve2d(icells, self._s,
                                   mode = 'same', boundary = 'wrap') != 0],
                [xp.asanyarray(1 << REG_S)],
                default = 0) +
            # C
            xp.select(
                [self._cells],
                [xp.asanyarray(1 << REG_C)],
                default = 0))
        #
        r = xp.left_shift(xp.asanyarray(xp.int64(1)), r)
        self._cells = xp.select(
            [(r & self._rule) != 0],
            [xp.asanyarray(xp.bool_(True))],
            default = xp.bool_(False))
    def entropy(self) -> float:
        mask = xp.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]], dtype = xp.uint64)
        icells = xp.select([self._cells], [xp.asarray(1)], default = 0)
        p_one : float = xp.select([signal.convolve2d(
            icells, mask,
            mode = 'same', boundary = 'wrap') != 0],
                          [xp.asanyarray(1)], default = 0).sum() / (self._height * self._width)
        p_zero = 1.0 - p_one
        try:
            return -(p_one * math.log(p_one) + p_zero * math.log(p_zero))
        except:
            return 0.0
    def sticky_rate(self) -> float:
        icells = xp.select([self._cells], [xp.asarray(1)], default = 0)
        r_n: float = ((signal.convolve2d(
            icells, self._n,
            mode = 'same', boundary = 'wrap') - icells) ** 2).sum() / (self._height * self._width)
        r_e: float = ((signal.convolve2d(
            icells, self._e,
            mode = 'same', boundary = 'wrap') - icells) ** 2).sum() / (self._height * self._width)
        r_w: float = ((signal.convolve2d(
            icells, self._w,
            mode = 'same', boundary = 'wrap') - icells) ** 2).sum() / (self._height * self._width)
        r_s: float = ((signal.convolve2d(
            icells, self._s,
            mode = 'same', boundary = 'wrap') - icells) ** 2).sum() / (self._height * self._width)
        r_c: float = xp.select(
            [self._prev_cells == self._cells],
            [xp.asanyarray(xp.uint64(1))],
            default = xp.uint64(0)).sum() / (self._height * self._width)
        return r_n * r_e * r_w * r_s * r_c

class AnimationGIF(object):
    def __init__(self, width: int, height: int, fps: float) -> None:
        self._fps = fps
        self.im_list : list[Any] = []
    def capture(self, bgr: Any) -> None:
        # BGR to RGB
        rgb = xp.copy(bgr)
        rgb[:,:,0] = bgr[:,:,2]
        rgb[:,:,1] = bgr[:,:,1]
        rgb[:,:,2] = bgr[:,:,0]
        im = Image.fromarray(rgb)
        self.im_list.append(im)
    def make_gif(self) -> None:
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = date + ".gif"
        duration_time = 1000 // self._fps
        print("duration:{}".format(duration_time))
        self.im_list[0].save(path,
                             save_all=True,
                             append_images=self.im_list[1:],
                             duration=duration_time, loop=0)
class Animation(object):
    def __init__(self, width: int, height: int, fps: float, outfile: Optional[str] = None) -> None:
        if not outfile:
            date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            outfile = date + '.mp4'
        fmt = cv2.VideoWriter_fourcc(*'avc1')
        self._writer = cv2.VideoWriter(outfile, fmt, fps,
                                       (width, height))
    def capture(self, bgr: Any) -> None:
        self._writer.write(bgr)
    def make_gif(self) -> None:
        self._writer.release()
    
def main() -> None:
    # --- Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type = int, default = 200,
                        help = 'Field width (default 200)')
    parser.add_argument('--height', type = int, default = 200,
                        help = 'Field height (default 200)')
    parser.add_argument('--size', type = int, default = 1,
                        help = 'Magnify cell size (default 1)')
    parser.add_argument('--rule', type = str, default = '',
                        help = 'Rule (default rule is random)')
    parser.add_argument('--loop', type = int, default = 1000,
                        help = 'loop count (default 1000)')
    parser.add_argument('--batch', action = 'store_true',
                        help = 'batch mode (run without graphics)')
    parser.add_argument('--animation', action = 'store_true',
                        help = 'animation')
    args = parser.parse_args()
    width = int(args.width)
    height = int(args.height)
    r = hex(int(random.random() * (1 << 32)) & ((1 << 32) - 1))
    rule = (
        int(args.rule, 16) if args.rule
        else int(random.random() * (1 << 32)) & ((1 << 32) - 1))
    loop = args.loop
    is_batch = args.batch
    # --- Calculate and display loop
    field = Field(width, height, rule)
    field.init_random()
    field.mask()
    bgr_img = field.get_current_bgr_image(args.size)
    if args.animation:
        animation = Animation(width, height, 100,
                              outfile = args.rule + '.mp4')
    waiting = 10
    for n in range(loop):
        field.update_cells()
        bgr_img = field.get_current_bgr_image(args.size)
        if is_batch:
            continue
        cv2.imshow("Ceullular Automata", bgr_img)
        key = cv2.waitKey(waiting)
        if key == ord("+"):
            waiting = max(10, waiting//2)
        if key == ord("-"):
            waiting = min(1000, waiting*2)
        if key == ord("q"):
            break
        if args.animation:
            animation.capture(bgr_img)
    print('{0:.08f} rule={1} sticky_rate={2:.08f} Entropy={3}'.format(
        field.sticky_rate() * field.entropy(),
        hex(rule),
        field.sticky_rate(), field.entropy()))
    if args.animation:
        print('dumping mp4 animation...')
        animation.make_gif()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
