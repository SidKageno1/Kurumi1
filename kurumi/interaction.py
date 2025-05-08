import pyautogui
from kurumi.movement import ScreenController
import logging

class UIInteractor:
    def __init__(self):
        self.screen = ScreenController()
        self.action_map = {
            0: self._click_element,
            1: self._type_text,
            2: self._scroll,
            3: self._press_key,
            4: self._drag_element
        }

    def _scroll(self, clicks: int, *_):
        """Прокрутка экрана"""
        try:
            self.screen.scroll(clicks)
            return True
        except Exception as e:
            logging.error(f"Ошибка прокрутки: {str(e)}")
            return False

    def _press_key(self, key: str, *_):
        """Нажатие клавиши"""
        try:
            self.screen.press_key(key)
            return True
        except Exception as e:
            logging.error(f"Ошибка нажатия клавиши: {str(e)}")
            return False

    def _drag_element(self, start_coords, end_coords):
        """Перетаскивание элемента"""
        try:
            self.screen.move_cursor(*start_coords)
            self.screen.mouse_down()
            self.screen.move_cursor(*end_coords)
            self.screen.mouse_up()
            return True
        except Exception as e:
            logging.error(f"Ошибка перетаскивания: {str(e)}")
            return False


    def _click_element(self, coords, *_):
        self.screen.move_cursor(*coords)
        self.screen.click()
        return True

    def _type_text(self, coords, text):
        self.screen.move_cursor(*coords)
        self.screen.click()
        self.screen.type_text(text)
        return True

