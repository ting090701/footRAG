const chatDiv = document.getElementById('chat');
const inputBox = document.getElementById('input');
const sendBtn = document.getElementById('send');

let isProcessing = false;

// Markdown 轉 HTML
function parseMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>')
    .replace(/(\d+\.\s)/g, '<br>$1')
    .replace(/(-\s)/g, '<br>• ');
}

function appendMessage(text, cls, isSystem = false) {
  const messageContainer = document.createElement('div');
  messageContainer.className = 'message-container';
  
  const p = document.createElement('p');
  
  if (cls === 'bot' || isSystem) {
    p.innerHTML = parseMarkdown(text);
  } else {
    p.textContent = text;
  }
  
  p.className = `msg ${cls}`;
  if (isSystem) {
    p.className += ' system';
  }
  
  messageContainer.appendChild(p);
  chatDiv.appendChild(messageContainer);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

// 移除「思考中」訊息
function removeThinkingMessage() {
  const lastChild = chatDiv.lastChild;
  // 檢查是否有正在思考的訊息，避免誤刪
  if (lastChild && lastChild.innerText.includes('…思考中…')) {
    chatDiv.removeChild(lastChild);
  }
}

async function sendMessage() {
  if (isProcessing) return;
  
  const question = inputBox.value.trim();
  if (!question) return;
  
  isProcessing = true;
  sendBtn.disabled = true;
  
  appendMessage(`你：${question}`, 'user');
  inputBox.value = '';
  appendMessage('…思考中…', 'bot');

  try {
    // ✨ 修改點 1: 網址改成 /chat (對應 server.py 的 @app.post("/chat"))
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      // ✨ 修改點 2: key 必須是 "message" (對應 server.py 的 UserRequest class)
      body: JSON.stringify({ message: question }) 
    });
    
    const data = await res.json();
    
    removeThinkingMessage();
    
    // ✨ 修改點 3: 後端回傳的 key 是 "reply" (對應 server.py 的 return {"reply": ...})
    if (data.reply) {
      appendMessage(`Bot：${data.reply}`, 'bot');
    } else {
      appendMessage(`Bot：${data.error || '發生未知錯誤'}`, 'bot');
    }
    
  } catch (e) {
    console.error(e);
    removeThinkingMessage();
    appendMessage('Bot：連線錯誤，請確認後端伺服器 (server.py) 已啟動。', 'bot');
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    inputBox.focus(); // 發送完自動聚焦回輸入框
  }
}

// 綁定事件
sendBtn.onclick = sendMessage;

inputBox.addEventListener('keypress', function(e) {
  if (e.key === 'Enter' && !isProcessing) {
    sendMessage();
  }
});

// 頁面載入初始化
document.addEventListener('DOMContentLoaded', function() {
  // 顯示歡迎訊息
  setTimeout(() => {
    appendMessage('系統：歡迎使用 FootAnalyzer 客服系統！請輸入您的問題。', 'bot', true);
  }, 500);
});