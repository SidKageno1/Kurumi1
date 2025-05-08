import pyautogui
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
import os
class ScreenController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self._configure_safety()
        logging.info(f"Screen controller initialized. Resolution: {self.screen_width}x{self.screen_height}")

    def _configure_safety(self):
        """Настройка параметров безопасности"""
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.15

    def move_cursor(self, x: int, y: int, duration: float = 0.1) -> bool:
        """Плавное перемещение курсора с обработкой ошибок"""
        try:
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            pyautogui.moveTo(x, y, duration=duration)
            logging.debug(f"Moved to ({x}, {y})")
            return True
        except Exception as e:
            logging.error(f"Move error: {str(e)}")
            return False

    def click(self, button: str = 'left', clicks: int = 1, **kwargs) -> bool:
        """Улучшенный метод клика с дополнительными параметрами"""
        try:
            pyautogui.click(button=button, clicks=clicks, **kwargs)
            logging.info(f"Clicked {button} {clicks} times")
            return True
        except pyautogui.FailSafeException:
            logging.critical("Fail-safe triggered!")
            raise
        except Exception as e:
            logging.error(f"Click error: {str(e)}")
            return False

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Ввод текста с контролем скорости"""
        try:
            pyautogui.write(text, interval=interval)
            logging.info(f"Typed: {text}")
            return True
        except Exception as e:
            logging.error(f"Typing error: {str(e)}")
            return False

    def press_key(self, key: str, presses: int = 1) -> bool:
        """Нажатие клавиши с проверкой допустимости"""
        valid_keys = ['esc', 'enter', 'tab', 'space', 'shift']
        if key.lower() not in valid_keys:
            logging.warning(f"Unsupported key: {key}")
            return False

        try:
            pyautogui.press(key, presses=presses)
            logging.info(f"Pressed {key} {presses} times")
            return True
        except Exception as e:
            logging.error(f"Key press error: {str(e)}")
            return False

    def find_image(self,
                   template_path: str,
                   confidence: float = 0.8,
                   region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int]]:
        """Поиск изображения на экране с обработкой ошибок"""
        try:
            if not os.path.exists(template_path):
                logging.error(f"Template file not found: {template_path}")
                return None

            screenshot = pyautogui.screenshot(region=region)
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)

            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= confidence:
                h, w = template.shape[:2]
                center_x = max_loc[0] + w // 2 + (region[0] if region else 0)
                center_y = max_loc[1] + h // 2 + (region[1] if region else 0)
                logging.info(f"Found {template_path} at ({center_x}, {center_y})")
                return (center_x, center_y)
            return None
        except Exception as e:
            logging.error(f"Image search error: {str(e)}")
            return None

    def scroll(self, clicks: int, smooth: bool = True) -> bool:
        """Прокрутка с опцией плавности"""
        try:
            if smooth:
                for _ in range(abs(clicks)):
                    pyautogui.scroll(clicks // abs(clicks))
                    pyautogui.sleep(0.05)
            else:
                pyautogui.scroll(clicks)

            logging.info(f"Scrolled {clicks} clicks")
            return True
        except Exception as e:
            logging.error(f"Scroll error: {str(e)}")
            return False

    def drag(self,
             start_x: int,
             start_y: int,
             end_x: int,
             end_y: int,
             duration: float = 0.5) -> bool:
        """Плавное перетаскивание с зажатой кнопкой"""
        try:
            self.move_cursor(start_x, start_y)
            pyautogui.mouseDown()
            self.move_cursor(end_x, end_y, duration=duration)
            pyautogui.mouseUp()
            logging.info(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
        except Exception as e:
            logging.error(f"Drag error: {str(e)}")
            return False