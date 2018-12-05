# План тестирования
---
## Содержение
1. [Введение](#intro)
2. [Объект тестирования](#test_object)
3. [Риски](#risk)
4. [Аспекты тестирования](#test_aspects) 
5. [Подходы к тестированию](#test_approaches) 
6. [Представление результатов](#tests_results) 
7. [Выводы](#conclusion) 
---
<a name="intro"></a>
## Введение
Данный документ содержит описание этапа тестирования библиотеки распознавания DocRecognition. Он производится командой, в первую очередь состоящей из разработчиков, пиров и заказчиков.
<a name="test_object"></a>
## Объект тестирования
Объект тестирования рассматирвается как с точки зрения отдельных модулей (например, компоненты загрузки изображенией, обработки изображений, получения результатов распознавания), так и системы в целом. Библиотека DocRecognition должна обладать следующими атрибутами качества, по [**ISO 25010**](https://www.iso.org/standard/35733.html):  
* функциональными:
 * функциональная корректность;
* производительности:
 * временные характеристики;
* удобства использования:
 * изучаемость (learnability);
* переносимости:
 * адаптируемость (adaptability);
 * устанавливаемость (installability).  

Более конретую информацию об атрибутах качества библиотеки распознавания DocRecognition  можно найти в разделе [аспекты тестирования](#test_approaches) текущего документа или в [SRS](https://github.com/AleshaShch/DocRecognition/blob/master/Documents/SRS.md) системы.  
<a name="risk"></a>
## Риски
Дополнительным условием, которое может повлиять на систему распознавания, является взаимодествие библиотеки с внешними программными модулями. 
<a name="test_aspects"></a>
## Аспекты тестирования
Согласно [SRS](https://github.com/AleshaShch/DocRecognition/blob/master/Documents/SRS.md) библиотеки распознавания неоходимо протестировать и оценить реализацию следующих функциональных требований:
1. *загрузка изображения*;
2. *выполнение распознавания изображения*;
3. *получение результатов распознавания изображения*.

А также нефункциональных требований:
* *произодительность*, распознание изображения должно осуществляться за время меньшее чем 1 секунда. 
* *удобство использования*, библиотека должна содержать документацию к исходному коду;
* *эффективность*, система должна распознавать информационные поля банковских карт со следующими процентными показателями качества: качество распознания номера карты - 70%, качество распознания срока действия карты -70%, качество распознания имени держателя карты - 55%;
* *переносимость*, Библиотека должна работать корректно на следующих операционных системах: Windows, Linux, Android. 

Требования удобство использования и переносимость являются характерными для такого типа ПО как библиотеки.
Также необходимо протестировать выполнение программы в командной строке.
<a name="test_approaches"></a>
## Подходы к тестированию
Тестирование библиотеки распознавания DocRecognition осуществляется на основе следующих принципов:
* По запуску кода на исполнение: *динамическое тестирование*;
* По доступу к коду и архитектуре приложения: *метод белого ящика*;
* По степени автоматизациии: *ручное тестирование*;
* По уровню детализации приложения: *системное тестирование*;
* По уровню функционального тестирования: *расширенное тестирование*;
* По принципам работы с приложением: *позитивное тестирование*;
* По природе приложения: *тестирование настольных приложений*;
* По времени тестирования: *альфа-тестирование*;
* По степени формализации: *тестирование на основе тест-кейсов*.

<a name="tests_results"></a>
## Представление результатов
Результаты тестирования представлены в отдельном [документе](/TestsResults). 
<a name="conclusion"></a>
## Выводы
Исходя из результатов тестирования, можно сказать, что программа соответствует требованием, описанным в SRS, за исключением внутреннего атрибута качества, эффективность, так как качетсва распознавания срока действия карты меньше 70%.   