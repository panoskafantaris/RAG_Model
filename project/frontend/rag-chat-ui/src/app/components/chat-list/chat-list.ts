import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatService } from '../../services/chat.service';
import { ChatSession } from '../../models/chat';
import { ListboxModule } from 'primeng/listbox';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-chat-list',
  standalone: true,
  templateUrl: './chat-list.html',
  styleUrl: './chat-list.scss',
  imports: [CommonModule, ListboxModule, ButtonModule],
})
export class ChatListComponent {
  sessions: ChatSession[] = [];
  selectedSession: ChatSession = null as any;

  constructor(private chatService: ChatService) {
    this.chatService.getSessions().subscribe(s => this.sessions = s);
  }

  createSession() {
    const title = `Chat ${this.sessions.length + 1}`;
    this.selectedSession = this.chatService.newSession(title);
  }

  onSelectSession(event: any) {
    this.selectedSession = event.value;
    this.chatService.setActiveSession(this.selectedSession);
  }
}
