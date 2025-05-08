import pyautogui
import os
import cv2
import logging
import numpy as np
import torch
from typing import List, Dict, Union
from ultralytics import YOLO


class InterfaceDetector:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = YOLO("yolov8n.pt").to(self.device)
            self.logger.info(f"Базовая модель YOLO загружена")
            self.custom_model = None  # Отключаем кастомную модель
        except Exception as e:
            self.logger.error(f"Ошибка: {str(e)}")
            raise RuntimeError("Не удалось инициализировать детектор")

        except Exception as e:
            self.logger.error(f"Критическая ошибка инициализации: {str(e)}")
            raise RuntimeError("Не удалось инициализировать детектор")

    def detect_elements(self, screenshot: np.ndarray) -> List[Dict[str, Union[str, float, list]]]:
        """Обнаружение элементов интерфейса"""
        try:
            # Конвертация в BGR для OpenCV
            img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

            # Детекция объектов
            results = []
            if self.model:
                base_results = self.model(img, verbose=False)[0]
                results.extend(self._process_yolo_results(base_results, "base"))
            if self.custom_model:
                custom_results = self.custom_model(img, verbose=False)[0]
                results.extend(self._process_yolo_results(custom_results, "custom"))

            return self._apply_nms(results)

        except Exception as e:
            self.logger.error(f"Ошибка детекции: {str(e)}")
            return []

    def _process_yolo_results(self, results, source: str):
        """Обработка результатов YOLO"""
        return [{
            "box": box.xyxy[0].cpu().numpy().tolist(),
            "confidence": box.conf.item(),
            "class_id": int(box.cls),
            "source": source
        } for box in results.boxes]

    def _apply_nms(self, boxes: List[dict], iou_threshold: float = 0.5) -> List[dict]:
        """Фильтрация пересекающихся боксов"""
        if not boxes:
            return []

        try:
            # Подготовка данных для NMS
            boxes_tensor = torch.tensor([
                [*box["box"], box["confidence"]]
                for box in boxes
            ])

            # Применение NMS
            indices = torch.ops.torchvision.nms(
                boxes_tensor[:, :4],
                boxes_tensor[:, 4],
                iou_threshold
            ).cpu().numpy()

            # Форматирование результата
            return [{
                "label": self.class_names[boxes[i]["class_id"]],
                "confidence": round(boxes[i]["confidence"], 2),
                "bbox": [round(coord) for coord in boxes[i]["box"]],
                "source": boxes[i]["source"]
            } for i in indices]

        except Exception as e:
            self.logger.error(f"Ошибка NMS: {str(e)}")
            return []

    def _merge_results(self,
                       general_results: torch.Tensor,
                       ui_results: torch.Tensor) -> List[Dict[str, Union[str, float, list]]]:
        """Объединение и фильтрация результатов детекции"""
        combined_boxes = []
        class_names = {}

        # Обработка результатов базовой модели
        if general_results:
            class_names.update(general_results.names)
            for box in general_results.boxes:
                combined_boxes.append({
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": box.conf.item(),
                    "class_id": int(box.cls),
                    "source": "base_model"
                })

        # Обработка результатов кастомной модели
        if ui_results:
            class_names.update(ui_results.names)
            for box in ui_results.boxes:
                combined_boxes.append({
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": box.conf.item(),
                    "class_id": int(box.cls),
                    "source": "ui_model"
                })

        # Применение NMS для устранения пересекающихся боксов
        filtered_boxes = self._apply_nms(combined_boxes)

        # Форматирование результатов
        return [{
            "label": class_names[box["class_id"]],
            "confidence": round(box["confidence"], 2),
            "bbox": [round(coord) for coord in box["box"]],
            "source": box["source"]
        } for box in filtered_boxes]

    def _apply_nms(self, boxes: List[dict], iou_threshold: float = 0.5) -> List[dict]:
        """Non-Maximum Suppression для фильтрации пересекающихся боксов"""
        if not boxes:
            return []

        # Конвертация в формат для NMS
        boxes_tensor = torch.tensor([b["box"] + [b["confidence"]] for b in boxes])

        # Применение NMS
        indices = torch.ops.torchvision.nms(
            boxes_tensor[:, :4],
            boxes_tensor[:, 4],
            iou_threshold
        ).cpu().numpy()

        return [boxes[i] for i in indices]


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = InterfaceDetector()
    screenshot = np.array(pyautogui.screenshot())

    elements = detector.detect_elements(screenshot)
    for elem in elements:
        print(f"Обнаружен {elem['label']} с уверенностью {elem['confidence']} в области {elem['bbox']}")