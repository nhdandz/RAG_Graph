"use client";

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/shared/Card';
import { Button } from '@/components/shared/Button';
import { Badge } from '@/components/shared/Badge';
import { getAllLogos, createLogo, updateLogo, deleteLogo, reorderLogos } from '@/lib/api/logos';
import type { Logo } from '@/lib/types/logo';

export default function AdminPage() {
  const [logos, setLogos] = useState<Logo[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingLogo, setEditingLogo] = useState<Logo | null>(null);
  const [newLogo, setNewLogo] = useState({
    name: '',
    imageUrl: '',
    websiteUrl: '',
    order: 0
  });

  useEffect(() => {
    loadLogos();
  }, []);

  const loadLogos = async () => {
    try {
      setLoading(true);
      const data = await getAllLogos();
      setLogos(data);
    } catch (error) {
      console.error('Failed to load logos:', error);
      alert('Không thể tải logos');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!newLogo.name || !newLogo.imageUrl) {
      alert('Vui lòng nhập tên và URL logo');
      return;
    }

    try {
      await createLogo({
        ...newLogo,
        order: logos.length + 1
      });

      setNewLogo({ name: '', imageUrl: '', websiteUrl: '', order: 0 });
      await loadLogos();
      alert('Đã tạo logo thành công!');
    } catch (error) {
      console.error('Failed to create logo:', error);
      alert('Không thể tạo logo');
    }
  };

  const handleUpdate = async (logo: Logo) => {
    try {
      await updateLogo(logo.id, {
        name: logo.name,
        imageUrl: logo.imageUrl,
        websiteUrl: logo.websiteUrl,
        active: logo.active,
        order: logo.order
      });

      setEditingLogo(null);
      await loadLogos();
      alert('Đã cập nhật logo thành công!');
    } catch (error) {
      console.error('Failed to update logo:', error);
      alert('Không thể cập nhật logo');
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Bạn có chắc muốn xóa logo này?')) return;

    try {
      await deleteLogo(id);
      await loadLogos();
      alert('Đã xóa logo thành công!');
    } catch (error) {
      console.error('Failed to delete logo:', error);
      alert('Không thể xóa logo');
    }
  };

  const handleToggleActive = async (logo: Logo) => {
    try {
      await updateLogo(logo.id, { active: !logo.active });
      await loadLogos();
    } catch (error) {
      console.error('Failed to toggle active:', error);
    }
  };

  return (
    <div className="min-h-screen bg-bg-primary p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold text-military-green-900 dark:text-military-green-400">
              🔐 Admin Panel
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Quản lý logos trường quân sự
            </p>
          </div>
          <a href="/">
            <Button variant="outline">
              ← Về trang chủ
            </Button>
          </a>
        </div>
      </div>

      <div className="max-w-7xl mx-auto space-y-6">
        {/* Create New Logo */}
        <Card variant="military" title="➕ Thêm Logo Mới">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium mb-2">Tên trường *</label>
              <input
                type="text"
                value={newLogo.name}
                onChange={(e) => setNewLogo({ ...newLogo, name: e.target.value })}
                placeholder="Học viện Kỹ thuật Quân sự"
                className="w-full px-4 py-2 border border-gray-300 dark:border-zinc-600 rounded-md focus:outline-none focus:border-military-green-600 bg-white dark:bg-zinc-700"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">URL Logo *</label>
              <input
                type="text"
                value={newLogo.imageUrl}
                onChange={(e) => setNewLogo({ ...newLogo, imageUrl: e.target.value })}
                placeholder="/static/logos/hvktqs.png"
                className="w-full px-4 py-2 border border-gray-300 dark:border-zinc-600 rounded-md focus:outline-none focus:border-military-green-600 bg-white dark:bg-zinc-700"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium mb-2">Website URL (optional)</label>
              <input
                type="text"
                value={newLogo.websiteUrl}
                onChange={(e) => setNewLogo({ ...newLogo, websiteUrl: e.target.value })}
                placeholder="http://mta.edu.vn"
                className="w-full px-4 py-2 border border-gray-300 dark:border-zinc-600 rounded-md focus:outline-none focus:border-military-green-600 bg-white dark:bg-zinc-700"
              />
            </div>
          </div>
          <Button onClick={handleCreate}>
            ➕ Tạo Logo
          </Button>
        </Card>

        {/* Logos List */}
        <Card variant="military" title="📋 Danh sách Logos">
          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin w-12 h-12 border-4 border-military-green-600 border-t-transparent rounded-full mx-auto"></div>
              <p className="mt-4 text-gray-500">Đang tải...</p>
            </div>
          ) : logos.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              Chưa có logo nào
            </div>
          ) : (
            <div className="space-y-4">
              {logos.map((logo) => (
                <div
                  key={logo.id}
                  className="border border-gray-200 dark:border-zinc-700 rounded-lg p-4 hover:border-military-green-600 transition-colors"
                >
                  {editingLogo?.id === logo.id ? (
                    /* Edit Mode */
                    <div className="space-y-3">
                      <input
                        type="text"
                        value={editingLogo.name}
                        onChange={(e) => setEditingLogo({ ...editingLogo, name: e.target.value })}
                        className="w-full px-3 py-2 border rounded-md"
                      />
                      <input
                        type="text"
                        value={editingLogo.imageUrl}
                        onChange={(e) => setEditingLogo({ ...editingLogo, imageUrl: e.target.value })}
                        className="w-full px-3 py-2 border rounded-md"
                      />
                      <input
                        type="text"
                        value={editingLogo.websiteUrl || ''}
                        onChange={(e) => setEditingLogo({ ...editingLogo, websiteUrl: e.target.value })}
                        className="w-full px-3 py-2 border rounded-md"
                        placeholder="Website URL"
                      />
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleUpdate(editingLogo)}>
                          💾 Lưu
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => setEditingLogo(null)}>
                          ❌ Hủy
                        </Button>
                      </div>
                    </div>
                  ) : (
                    /* View Mode */
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-semibold text-lg">{logo.name}</h3>
                          <Badge
                            text={logo.active ? 'Active' : 'Inactive'}
                            color={logo.active ? 'green' : 'gray'}
                          />
                          <span className="text-sm text-gray-500">Order: {logo.order}</span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {logo.imageUrl}
                        </p>
                        {logo.websiteUrl && (
                          <p className="text-sm text-blue-600">
                            🔗 {logo.websiteUrl}
                          </p>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant={logo.active ? 'outline' : 'primary'}
                          onClick={() => handleToggleActive(logo)}
                        >
                          {logo.active ? '👁️ Hide' : '👁️ Show'}
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => setEditingLogo(logo)}
                        >
                          ✏️ Edit
                        </Button>
                        <Button
                          size="sm"
                          variant="danger"
                          onClick={() => handleDelete(logo.id)}
                        >
                          🗑️
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
