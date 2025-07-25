# 🎤 Фур'є-Караоке

**Фур’є-Караоке** — це вебдодаток, який дозволяє завантажити аудіофайл, а потім "перевести" його у синусоїди за допомогою **Фур’є-перетворення**. Ти можеш залишити тільки обмежену кількість гармонік — і почути, як звучить твій голос, зібраний з хвиль!

## 🔍 Як це працює?
1. 🎙️ **Запис або завантаження звуку** — через мікрофон або `.wav` файл.
2. 📊 **Фур’є-перетворення (FFT)** — розбиває аудіосигнал на частоти.
3. ✂️ **Обрізання гармонік** — залишаємо лише `N` найсильніших (ти вибираєш).
4. 🛠️ **Інверсія (IFFT)** — будуємо назад сигнал тільки з цих гармонік.
5. 🎧 **Порівняння** — слухаєш, як змінюється звучання:
   - Оригінал vs Реконструйований
   - Спектр частот
   - Графік сигналів у часі

## 🧠 Що це показує?
> "Будь-який звук — це комбінація синусоїд."

