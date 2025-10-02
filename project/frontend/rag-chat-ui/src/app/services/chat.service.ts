import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject } from 'rxjs';
import { ChatSession, ChatMessage } from '../models/chat';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = 'http://127.0.0.1:8000';
  private sessions: ChatSession[] = [];
  private sessions$ = new BehaviorSubject<ChatSession[]>([]);
  private activeSession$ = new BehaviorSubject<ChatSession | undefined>(undefined);

  constructor(private http: HttpClient) {}

  getSessions() {
    return this.sessions$.asObservable();
  }

  getActiveSession() {
    return this.activeSession$.asObservable();
  }

  setActiveSession(session: ChatSession) {
    this.activeSession$.next(session);
  }

  newSession(title: string) {
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title,
      messages: []
    };
    this.sessions.push(session);
    this.sessions$.next(this.sessions);
    this.setActiveSession(session);
    return session;
  }

  sendMessage(sessionId: string, message: string) {
    const session = this.sessions.find(s => s.id === sessionId);
    if (!session) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: message,
      timestamp: new Date()
    };
    session.messages.push(userMessage);

    this.http.post<{ answer: string }>(`${this.apiUrl}/chat`, { message })
      .subscribe(res => {
        const botMessage: ChatMessage = {
          role: 'assistant',
          content: res.answer,
          timestamp: new Date()
        };
        session.messages.push(botMessage);
        this.sessions$.next(this.sessions);
        this.setActiveSession(session);
      });
  }

  uploadFile(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/upload`, formData);
  }
}
